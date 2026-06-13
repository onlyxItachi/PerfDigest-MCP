#pragma once
// Stub of the GAFIME public interface, reconstructed from kernels.cu usage so
// the kernels compile into a standalone profiling target for perfdigest
// fixtures. Constant values only need to be distinct and ordered where the
// kernels range-check them; they are not the canonical GAFIME values.

#include <cstdint>
#include <cstring>

#define GAFIME_API

// Error codes
#define GAFIME_SUCCESS                   0
#define GAFIME_ERROR_INVALID_ARGS       -1
#define GAFIME_ERROR_OUT_OF_MEMORY      -2
#define GAFIME_ERROR_KERNEL_FAILED      -3
#define GAFIME_ERROR_CUDA_NOT_AVAILABLE -4

#define GAFIME_MAX_FEATURES 5

// Unary operators
#define GAFIME_OP_IDENTITY 0
#define GAFIME_OP_LOG      1
#define GAFIME_OP_EXP      2
#define GAFIME_OP_SQRT     3
#define GAFIME_OP_TANH     4
#define GAFIME_OP_SIGMOID  5
#define GAFIME_OP_SQUARE   6
#define GAFIME_OP_NEGATE   7
#define GAFIME_OP_ABS      8
#define GAFIME_OP_INVERSE  9
#define GAFIME_OP_CUBE     10

// Interaction combiners
#define GAFIME_INTERACT_MULT 0
#define GAFIME_INTERACT_ADD  1
#define GAFIME_INTERACT_SUB  2
#define GAFIME_INTERACT_DIV  3
#define GAFIME_INTERACT_MAX  4
#define GAFIME_INTERACT_MIN  5

// Candidate kinds (continuous + time-series)
#define GAFIME_CANDIDATE_CONTINUOUS      0
#define GAFIME_CANDIDATE_TS_LAG          1
#define GAFIME_CANDIDATE_TS_DELTA        2
#define GAFIME_CANDIDATE_TS_VELOCITY     3
#define GAFIME_CANDIDATE_TS_ACCELERATION 4
#define GAFIME_CANDIDATE_TS_ROLLING_SUM  5
#define GAFIME_CANDIDATE_TS_ROLLING_MEAN 6
#define GAFIME_CANDIDATE_TS_ROLLING_STD  7

// Discrete candidate kinds — must stay contiguous in this order; the host
// validators range-check [SOFT_THRESHOLD, VALUE_IN_SOFT_RECTANGLE].
#define GAFIME_DISCRETE_SOFT_THRESHOLD          0
#define GAFIME_DISCRETE_SOFT_INTERVAL           1
#define GAFIME_DISCRETE_VALUE_GATED_THRESHOLD   2
#define GAFIME_DISCRETE_SOFT_RECTANGLE          3
#define GAFIME_DISCRETE_VALUE_IN_SOFT_RECTANGLE 4

#define GAFIME_DISCRETE_DIRECTION_LE 0

// Stats layout: 12 floats per candidate (6 train + 6 val)
#define GAFIME_STATS_SIZE     12
#define GAFIME_STAT_TRAIN_N   0
#define GAFIME_STAT_TRAIN_SX  1
#define GAFIME_STAT_TRAIN_SY  2
#define GAFIME_STAT_TRAIN_SXX 3
#define GAFIME_STAT_TRAIN_SYY 4
#define GAFIME_STAT_TRAIN_SXY 5

// Selection score layout: 4 floats per candidate
#define GAFIME_SELECTION_SCORE_SIZE         4
#define GAFIME_SELECTION_MUTUAL_INFO        0
#define GAFIME_SELECTION_VARIANCE_REDUCTION 1
#define GAFIME_SELECTION_RESIDUAL_ABS_CORR  2
#define GAFIME_SELECTION_RESIDUAL_R2_GAIN   3

typedef void* GafimeBucket;
typedef void* GafimeCudaMatrix;

extern "C" {
GAFIME_API int gafime_cuda_available(void);
GAFIME_API int gafime_bucket_free(GafimeBucket bucket);
GAFIME_API int gafime_cuda_matrix_free(GafimeCudaMatrix matrix_handle);
GAFIME_API int gafime_cuda_matrix_alloc(
    int n_samples, int n_features, int max_batch_size, GafimeCudaMatrix* matrix_out);
GAFIME_API int gafime_cuda_matrix_upload(
    GafimeCudaMatrix matrix_handle, const float* h_X_colmajor, const float* h_y,
    const uint8_t* h_mask, const float* h_means);
GAFIME_API int gafime_cuda_matrix_compute_batch(
    GafimeCudaMatrix matrix_handle, const int* h_batch_indices, int arity,
    int batch_size, int val_fold_id, float* h_stats_batch);
}
