#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

Matrix* matrix_create(int rows, int cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) return NULL;
    
    m->rows = rows;
    m->cols = cols;
    m->data = (double*)calloc(rows * cols, sizeof(double));
    
    if (!m->data) {
        free(m);
        return NULL;
    }
    
    return m;
}

void matrix_free(Matrix* m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

void matrix_set(Matrix* m, int row, int col, double value) {
    if (m && row >= 0 && row < m->rows && col >= 0 && col < m->cols) {
        m->data[row * m->cols + col] = value;
    }
}

double matrix_get(Matrix* m, int row, int col) {
    if (m && row >= 0 && row < m->rows && col >= 0 && col < m->cols) {
        return m->data[row * m->cols + col];
    }
    return 0.0;
}

void matrix_randomize(Matrix* m) {
    if (!m) return;
    
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = (double)rand() / RAND_MAX;
    }
}

void matrix_zeros(Matrix* m) {
    if (!m) return;
    
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = 0.0;
    }
}

void matrix_print(Matrix* m) {
    if (!m) return;
    
    printf("Matrix (%d x %d):\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%8.4f ", matrix_get(m, i, j));
        }
        printf("\n");
    }
}

// Naive matrix multiplication: C = A * B
// Triple nested loop - most straightforward implementation
int matrix_multiply_naive(Matrix* A, Matrix* B, Matrix* C) {
    // Check dimensions: A (M x N) * B (N x P) = C (M x P)
    if (!A || !B || !C) return -1;
    if (A->cols != B->rows) return -1;
    if (C->rows != A->rows || C->cols != B->cols) return -1;
    
    int M = A->rows;
    int N = A->cols;
    int P = B->cols;
    
    // Naive triple loop: O(M*N*P)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                // C[i][j] += A[i][k] * B[k][j]
                sum += matrix_get(A, i, k) * matrix_get(B, k, j);
            }
            matrix_set(C, i, j, sum);
        }
    }
    
    return 0;
}

// Optimized matrix multiplication using transposed B for better cache locality
int matrix_multiply_transpose(Matrix* A, Matrix* B, Matrix* C) {
    // Check dimensions: A (M x N) * B (N x P) = C (M x P)
    if (!A || !B || !C) return -1;
    if (A->cols != B->rows) return -1;
    if (C->rows != A->rows || C->cols != B->cols) return -1;
    
    int M = A->rows;
    int N = A->cols;
    int P = B->cols;
    
    // Create transposed matrix of B (B_T is P x N)
    Matrix* B_T = matrix_create(P, N);
    if (!B_T) return -1;
    
    // Transpose B into B_T
    for (int i = 0; i < B->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            matrix_set(B_T, j, i, matrix_get(B, i, j));
        }
    }
    
    // Multiply A * B using transposed B
    // This makes inner loop access contiguous memory in both matrices
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                // A row access is contiguous, B_T row access is contiguous
                sum += matrix_get(A, i, k) * matrix_get(B_T, j, k);
            }
            matrix_set(C, i, j, sum);
        }
    }
    
    matrix_free(B_T);
    return 0;
}

// Get cache line size from system
// Returns 64 as default if detection fails
int get_cache_line_size(void) {
    long cache_line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (cache_line_size <= 0) {
        // Default to 64 bytes (most common)
        cache_line_size = 64;
    }
    return (int)cache_line_size;
}

// Cache-blocked matrix multiplication using tiling
// Optimizes for cache line size to maximize data reuse
int matrix_multiply_blocked(Matrix* A, Matrix* B, Matrix* C) {
    // Check dimensions: A (M x N) * B (N x P) = C (M x P)
    if (!A || !B || !C) return -1;
    if (A->cols != B->rows) return -1;
    if (C->rows != A->rows || C->cols != B->cols) return -1;
    
    int M = A->rows;
    int N = A->cols;
    int P = B->cols;
    
    // Get cache line size and calculate block size
    int cache_line_size = get_cache_line_size();
    // doubles are 8 bytes, so elements per cache line = cache_line_size / 8
    int elements_per_line = cache_line_size / sizeof(double);
    
    // Use block size that fits well in cache
    // A common approach: block size such that 3 blocks fit in L1 cache
    // Assuming 32KB L1 cache: 3 * BLOCK^2 * 8 bytes <= 32KB
    // BLOCK <= sqrt(32768 / 24) ~= 37, so we use 32 or 64
    int BLOCK = elements_per_line;
    if (BLOCK < 16) BLOCK = 16;
    if (BLOCK > 64) BLOCK = 64;
    
    // Initialize C to zeros first
    matrix_zeros(C);
    
    // Blocked/tiled matrix multiplication
    // Process sub-blocks that fit in cache for better locality
    for (int ii = 0; ii < M; ii += BLOCK) {
        int i_max = (ii + BLOCK < M) ? ii + BLOCK : M;
        
        for (int jj = 0; jj < P; jj += BLOCK) {
            int j_max = (jj + BLOCK < P) ? jj + BLOCK : P;
            
            for (int kk = 0; kk < N; kk += BLOCK) {
                int k_max = (kk + BLOCK < N) ? kk + BLOCK : N;
                
                // Multiply current blocks: C[ii:i_max][jj:j_max] += 
                //   A[ii:i_max][kk:k_max] * B[kk:k_max][jj:j_max]
                for (int i = ii; i < i_max; i++) {
                    for (int j = jj; j < j_max; j++) {
                        double sum = matrix_get(C, i, j);
                        
                        // Inner loop - process cache-friendly block
                        for (int k = kk; k < k_max; k++) {
                            sum += matrix_get(A, i, k) * matrix_get(B, k, j);
                        }
                        
                        matrix_set(C, i, j, sum);
                    }
                }
            }
        }
    }
    
    return 0;
}
