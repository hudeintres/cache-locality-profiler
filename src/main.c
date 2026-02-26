#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "profiler.h"
#include "matrix.h"

// Test matrix multiplication with profiling
void test_matrix_multiplication(int size, int block_size, Profiler* profiler) {
    char label[64];
    
    snprintf(label, sizeof(label), "matrix_create_%dx%d", size, size);
    profiler_start(profiler, label);
    Matrix* A = matrix_create(size, size);
    Matrix* B = matrix_create(size, size);
    Matrix* C_naive = matrix_create(size, size);
    Matrix* C_transpose = matrix_create(size, size);
    Matrix* C_blocked = matrix_create(size, size);
    profiler_end(profiler, label);
    
    if (!A || !B || !C_naive || !C_transpose || !C_blocked) {
        fprintf(stderr, "Failed to allocate matrices\n");
        return;
    }
    
    // Initialize matrices
    snprintf(label, sizeof(label), "matrix_init_%dx%d", size, size);
    profiler_start(profiler, label);
    matrix_randomize(A);
    matrix_randomize(B);
    matrix_zeros(C_naive);
    matrix_zeros(C_transpose);
    matrix_zeros(C_blocked);
    profiler_end(profiler, label);
    
    // Perform naive multiplication
    snprintf(label, sizeof(label), "matrix_multiply_naive_%dx%d", size, size);
    profiler_start(profiler, label);
    int result_naive = matrix_multiply_naive(A, B, C_naive);
    profiler_end(profiler, label);
    
    if (result_naive != 0) {
        fprintf(stderr, "Naive matrix multiplication failed\n");
    }
    
    // Perform transpose-optimized multiplication
    snprintf(label, sizeof(label), "matrix_multiply_transpose_%dx%d", size, size);
    profiler_start(profiler, label);
    int result_transpose = matrix_multiply_transpose(A, B, C_transpose);
    profiler_end(profiler, label);
    
    if (result_transpose != 0) {
        fprintf(stderr, "Transpose-optimized matrix multiplication failed\n");
    }
    
    // Perform cache-blocked multiplication (block_size pre-calculated)
    snprintf(label, sizeof(label), "matrix_multiply_blocked_%dx%d", size, size);
    profiler_start(profiler, label);
    int result_blocked = matrix_multiply_blocked(A, B, C_blocked, block_size);
    profiler_end(profiler, label);
    
    if (result_blocked != 0) {
        fprintf(stderr, "Cache-blocked matrix multiplication failed\n");
    }
    
    // Sanity check: compare results across methods
    double max_diff_transpose = 0.0;
    double max_diff_blocked = 0.0;
    for (int i = 0; i < size * size; i++) {
        double naive_val = C_naive->data[i];
        double transpose_val = C_transpose->data[i];
        double blocked_val = C_blocked->data[i];
        
        double diff_transpose = fabs(naive_val - transpose_val);
        double diff_blocked = fabs(naive_val - blocked_val);
        
        if (diff_transpose > max_diff_transpose) {
            max_diff_transpose = diff_transpose;
        }
        if (diff_blocked > max_diff_blocked) {
            max_diff_blocked = diff_blocked;
        }
    }
    
    const double tolerance = 1e-9;
    if (max_diff_transpose > tolerance || max_diff_blocked > tolerance) {
        fprintf(stderr, "Sanity check failed for size %dx%d: max diff transpose=%.6e, blocked=%.6e\n",
                size, size, max_diff_transpose, max_diff_blocked);
    }
    
    // Cleanup
    snprintf(label, sizeof(label), "matrix_free_%dx%d", size, size);
    profiler_start(profiler, label);
    matrix_free(A);
    matrix_free(B);
    matrix_free(C_naive);
    matrix_free(C_transpose);
    matrix_free(C_blocked);
    profiler_end(profiler, label);
}

int main(int argc, char* argv[]) {
    Profiler profiler;
    profiler_init(&profiler);
    
    printf("Matrix Multiplication Profiling\n");
    printf("================================\n\n");
    
    // Get cache info BEFORE starting any timers
    int cache_line_size = get_cache_line_size();
    int l1_cache_size = get_l1_cache_size();
    
    // Calculate optimal block size for cache-blocked multiplication
    // BLOCK^2 * 4 * sizeof(double) <= L1_cache_size (accounting for 4 arrays in working set)
    int max_elements = l1_cache_size / (4 * sizeof(double));
    int block_size = 1;
    while (block_size * block_size <= max_elements && block_size < 128) {
        block_size *= 2;
    }
    block_size /= 2;
    if (block_size < 16) block_size = 16;
    
    printf("System cache line size: %d bytes\n", cache_line_size);
    printf("L1 data cache size: %d bytes (%d KB)\n", l1_cache_size, l1_cache_size / 1024);
    printf("Elements per cache line (double): %zu\n", cache_line_size / sizeof(double));
    printf("Optimal block size for tiling: %d x %d\n\n", block_size, block_size);
    
    // Test with different matrix sizes
    int sizes[] = {64, 128, 256, 512};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    // Optional size from command line
    int extra_size = 0;
    if (argc > 1) {
        char* endptr = NULL;
        long value = strtol(argv[1], &endptr, 10);
        if (endptr == argv[1] || *endptr != '\0' || value <= 0) {
            fprintf(stderr, "Invalid size '%s'. Using default sizes only.\n", argv[1]);
        } else if (value > 4096) {
            fprintf(stderr, "Requested size %ld too large (max 4096). Using default sizes only.\n", value);
        } else {
            extra_size = (int)value;
            printf("Adding user-specified size: %dx%d\n", extra_size, extra_size);
        }
    }
    
    // Number of iterations for averaging
    int iterations = 3;
    
    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        printf("Testing %dx%d matrix multiplication (%d iterations)...\n", 
               size, size, iterations);
        
        for (int i = 0; i < iterations; i++) {
            test_matrix_multiplication(size, block_size, &profiler);
        }
    }
    
    if (extra_size > 0) {
        printf("Testing %dx%d matrix multiplication (%d iterations)...\n", 
               extra_size, extra_size, iterations);
        
        for (int i = 0; i < iterations; i++) {
            test_matrix_multiplication(extra_size, block_size, &profiler);
        }
    }
    
    // Print and save results
    profiler_print_results(&profiler);
    profiler_save_results(&profiler, "profile_results.csv");
    
    return 0;
}
