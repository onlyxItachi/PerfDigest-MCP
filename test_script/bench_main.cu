// Minimal benchmark harness for the GAFIME kernels. Exists so that
//   ncu --set full -o report.ncu-rep gafime_bench.exe
// captures real kernel launches (gafime_global_continuous_kernel at arity
// 1/2/3/5) as a perfdigest test fixture.
#include "interfaces.h"
#include <cstdio>
#include <vector>
#include <random>

int main() {
    if (!gafime_cuda_available()) {
        fprintf(stderr, "no CUDA device available\n");
        return 1;
    }

    const int n_samples = 1 << 20;  // 1M rows
    const int n_features = 32;
    const int batch_size = 128;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> X(static_cast<size_t>(n_samples) * n_features);
    std::vector<float> y(n_samples);
    std::vector<uint8_t> mask(n_samples);
    std::vector<float> means(n_features, 0.0f);
    for (auto& v : X) v = dist(rng);
    for (int i = 0; i < n_samples; ++i) {
        y[i] = dist(rng);
        mask[i] = static_cast<uint8_t>(i % 5);  // 5-fold split
    }
    for (int f = 0; f < n_features; ++f) {
        double sum = 0.0;
        for (int i = 0; i < n_samples; ++i)
            sum += X[static_cast<size_t>(f) * n_samples + i];
        means[f] = static_cast<float>(sum / n_samples);
    }

    GafimeCudaMatrix matrix = nullptr;
    int rc = gafime_cuda_matrix_alloc(n_samples, n_features, batch_size, &matrix);
    if (rc != GAFIME_SUCCESS) { fprintf(stderr, "alloc failed: %d\n", rc); return 1; }
    rc = gafime_cuda_matrix_upload(matrix, X.data(), y.data(), mask.data(), means.data());
    if (rc != GAFIME_SUCCESS) { fprintf(stderr, "upload failed: %d\n", rc); return 1; }

    std::vector<float> stats(static_cast<size_t>(batch_size) * GAFIME_STATS_SIZE);
    std::uniform_int_distribution<int> fdist(0, n_features - 1);

    for (int arity : {1, 2, 3, 5}) {
        std::vector<int> indices(static_cast<size_t>(batch_size) * arity);
        for (auto& ix : indices) ix = fdist(rng);
        rc = gafime_cuda_matrix_compute_batch(
            matrix, indices.data(), arity, batch_size, /*val_fold_id=*/0, stats.data());
        if (rc != GAFIME_SUCCESS) {
            fprintf(stderr, "compute_batch arity %d failed: %d\n", arity, rc);
            return 1;
        }
        fprintf(stderr, "arity %d ok: train_n[0]=%.0f val_n[0]=%.0f\n",
                arity, stats[0], stats[6]);
    }

    gafime_cuda_matrix_free(matrix);
    fprintf(stderr, "benchmark complete\n");
    return 0;
}
