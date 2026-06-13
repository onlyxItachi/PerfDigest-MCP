// Benchmark harness for the GAFIME kernels. Exists so that
//   ncu --set full -o report.ncu-rep gafime_bench.exe
// captures a varied set of real kernel launches as a perfdigest fixture.
//
// Two distinct kernels, several launch configs each:
//   * gafime_global_continuous_kernel<ARITY>  (matrix path) at arity 1..5
//   * gafime_batched_kernel<ARITY>            (bucket path) continuous + a
//     time-series rolling-window transform, with operators + interactions
// so the fixture exercises memory-bound and more compute-heavy profiles.
#include "interfaces.h"
#include <cstdio>
#include <vector>
#include <random>

static std::mt19937 g_rng(42);

static void check(int rc, const char* what) {
    if (rc != GAFIME_SUCCESS) {
        fprintf(stderr, "%s failed: %d\n", what, rc);
        std::exit(1);
    }
}

// ---- matrix path: gafime_global_continuous_kernel at every arity ----------
static void run_matrix_path() {
    const int n_samples = 1 << 20;  // 1M rows
    const int n_features = 32;
    const int batch_size = 128;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> X(static_cast<size_t>(n_samples) * n_features);
    std::vector<float> y(n_samples);
    std::vector<uint8_t> mask(n_samples);
    std::vector<float> means(n_features, 0.0f);
    for (auto& v : X) v = dist(g_rng);
    for (int i = 0; i < n_samples; ++i) { y[i] = dist(g_rng); mask[i] = static_cast<uint8_t>(i % 5); }
    for (int f = 0; f < n_features; ++f) {
        double s = 0.0;
        for (int i = 0; i < n_samples; ++i) s += X[static_cast<size_t>(f) * n_samples + i];
        means[f] = static_cast<float>(s / n_samples);
    }

    GafimeCudaMatrix matrix = nullptr;
    check(gafime_cuda_matrix_alloc(n_samples, n_features, batch_size, &matrix), "matrix_alloc");
    check(gafime_cuda_matrix_upload(matrix, X.data(), y.data(), mask.data(), means.data()), "matrix_upload");

    std::vector<float> stats(static_cast<size_t>(batch_size) * GAFIME_STATS_SIZE);
    std::uniform_int_distribution<int> fdist(0, n_features - 1);
    for (int arity : {1, 2, 3, 4, 5}) {
        std::vector<int> idx(static_cast<size_t>(batch_size) * arity);
        for (auto& ix : idx) ix = fdist(g_rng);
        check(gafime_cuda_matrix_compute_batch(matrix, idx.data(), arity, batch_size, 0, stats.data()),
              "matrix_compute_batch");
        fprintf(stderr, "matrix arity %d ok: train_n[0]=%.0f\n", arity, stats[0]);
    }
    gafime_cuda_matrix_free(matrix);
}

// ---- bucket path: gafime_batched_kernel (ops + interactions + TS) ---------
static void run_bucket_path() {
    const int n_samples = 1 << 19;   // different launch config than matrix path
    const int n_features = GAFIME_MAX_FEATURES;  // bucket holds <=5 feature columns
    const int batch_size = 64;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    GafimeBucket bucket = nullptr;
    check(gafime_bucket_alloc(n_samples, n_features, &bucket), "bucket_alloc");

    std::vector<float> col(n_samples), y(n_samples);
    std::vector<uint8_t> mask(n_samples);
    for (int f = 0; f < n_features; ++f) {
        for (int i = 0; i < n_samples; ++i) col[i] = dist(g_rng);
        check(gafime_bucket_upload_feature(bucket, f, col.data()), "upload_feature");
    }
    for (int i = 0; i < n_samples; ++i) { y[i] = dist(g_rng); mask[i] = static_cast<uint8_t>(i % 5); }
    check(gafime_bucket_upload_target(bucket, y.data()), "upload_target");
    check(gafime_bucket_upload_mask(bucket, mask.data()), "upload_mask");

    std::uniform_int_distribution<int> fdist(0, n_features - 1);
    std::vector<float> stats(static_cast<size_t>(batch_size) * GAFIME_STATS_SIZE);

    auto run = [&](const char* label, int arity, int kind, int op_a, int op_b,
                   int interact, int window) {
        std::vector<int> kinds(batch_size, kind);
        std::vector<int> idx(static_cast<size_t>(batch_size) * arity);
        std::vector<int> ops(static_cast<size_t>(batch_size) * arity);
        std::vector<int> inter(static_cast<size_t>(batch_size) * (arity - 1));
        std::vector<int> ts(static_cast<size_t>(batch_size) * 4);
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < arity; ++s) {
                idx[b * arity + s] = fdist(g_rng);
                ops[b * arity + s] = (s == 0) ? op_a : op_b;
            }
            for (int s = 0; s < arity - 1; ++s) inter[b * (arity - 1) + s] = interact;
            ts[b * 4 + 0] = 1;        // lag
            ts[b * 4 + 1] = window;   // rolling window
            ts[b * 4 + 2] = 0;
            ts[b * 4 + 3] = 0;
        }
        check(gafime_bucket_compute_batch(bucket, kinds.data(), idx.data(), ops.data(),
                                          inter.data(), ts.data(), arity, batch_size, 0, stats.data()),
              label);
        fprintf(stderr, "bucket %s ok: train_n[0]=%.0f\n", label, stats[0]);
    };

    run("continuous arity2 log*id", 2, GAFIME_CANDIDATE_CONTINUOUS,
        GAFIME_OP_LOG, GAFIME_OP_IDENTITY, GAFIME_INTERACT_MULT, 1);
    run("continuous arity3 sq+abs", 3, GAFIME_CANDIDATE_CONTINUOUS,
        GAFIME_OP_SQUARE, GAFIME_OP_ABS, GAFIME_INTERACT_ADD, 1);
    run("rolling_mean arity2 sub",  2, GAFIME_CANDIDATE_TS_ROLLING_MEAN,
        GAFIME_OP_IDENTITY, GAFIME_OP_IDENTITY, GAFIME_INTERACT_SUB, 8);

    gafime_bucket_free(bucket);
}

int main() {
    if (!gafime_cuda_available()) {
        fprintf(stderr, "no CUDA device available\n");
        return 1;
    }
    run_matrix_path();
    run_bucket_path();
    fprintf(stderr, "benchmark complete\n");
    return 0;
}
