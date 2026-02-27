#ifndef MATRIX_H
#define MATRIX_H

#ifdef __cplusplus
extern "C" {
#endif

// Matrix structure
typedef struct {
    double* data;
    int rows;
    int cols;
} Matrix;

// Create a matrix with given dimensions
Matrix* matrix_create(int rows, int cols);

// Free matrix memory
void matrix_free(Matrix* m);

// Set matrix element
void matrix_set(Matrix* m, int row, int col, double value);

// Get matrix element
double matrix_get(Matrix* m, int row, int col);

// Initialize matrix with random values
void matrix_randomize(Matrix* m);

// Initialize matrix with zeros
void matrix_zeros(Matrix* m);

// Print matrix (for debugging)
void matrix_print(Matrix* m);

// Naive matrix multiplication: C = A * B
// Returns 0 on success, -1 on dimension mismatch
int matrix_multiply_naive(Matrix* A, Matrix* B, Matrix* C);

// Optimized matrix multiplication using transposed B
// Returns 0 on success, -1 on dimension mismatch
int matrix_multiply_transpose(Matrix* A, Matrix* B, Matrix* C);

// Cache-blocked matrix multiplication using tiling
// block_size: size of sub-blocks to process (0 = auto-detect optimal)
// Returns 0 on success, -1 on dimension mismatch
int matrix_multiply_blocked(Matrix* A, Matrix* B, Matrix* C, int block_size);

// Get cache line size (returns 64 as default if detection fails)
int get_cache_line_size(void);

// Get L1 data cache size in bytes
// Returns 32768 (32KB) as default if detection fails
int get_l1_cache_size(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// C++11 concurrent matrix multiplication (row-parallel using std::thread)
int matrix_multiply_naive_parallel(Matrix* A, Matrix* B, Matrix* C, int num_threads);
int matrix_multiply_transpose_parallel(Matrix* A, Matrix* B, Matrix* C, int num_threads);
int matrix_multiply_blocked_parallel(Matrix* A, Matrix* B, Matrix* C, int block_size, int num_threads);
#endif

#endif // MATRIX_H
