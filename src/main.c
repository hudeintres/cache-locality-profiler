#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "profiler.h"
#include "matrix.h"

// Test matrix multiplication with profiling
void test_matrix_multiplication(int size, Profiler* profiler) {
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
    
    // Perform cache-blocked multiplication
    snprintf(label, sizeof(label), "matrix_multiply_blocked_%dx%d", size, size);
    profiler_start(profiler, label);
    int result_blocked = matrix_multiply_blocked(A, B, C_blocked);
    profiler_end(profiler, label);
    
    if (result_blocked != 0) {
        fprintf(stderr, "Cache-blocked matrix multiplication failed\n");
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
    
    // Print cache line size info
    int cache_line_size = get_cache_line_size();
    printf("System cache line size: %d bytes\n", cache_line_size);
    printf("Elements per cache line (double): %zu\n\n", cache_line_size / sizeof(double));
    
    // Test with different matrix sizes
    int sizes[] = {64, 128, 256, 512};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    // Number of iterations for averaging
    int iterations = 3;
    
    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        printf("Testing %dx%d matrix multiplication (%d iterations)...\n", 
               size, size, iterations);
        
        for (int i = 0; i < iterations; i++) {
            test_matrix_multiplication(size, &profiler);
        }
    }
    
    // Print and save results
    profiler_print_results(&profiler);
    profiler_save_results(&profiler, "profile_results.csv");
    
    return 0;
}
