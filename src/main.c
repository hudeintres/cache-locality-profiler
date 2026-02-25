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
    Matrix* C = matrix_create(size, size);
    profiler_end(profiler, label);
    
    if (!A || !B || !C) {
        fprintf(stderr, "Failed to allocate matrices\n");
        return;
    }
    
    // Initialize matrices
    snprintf(label, sizeof(label), "matrix_init_%dx%d", size, size);
    profiler_start(profiler, label);
    matrix_randomize(A);
    matrix_randomize(B);
    matrix_zeros(C);
    profiler_end(profiler, label);
    
    // Perform naive multiplication
    snprintf(label, sizeof(label), "matrix_multiply_naive_%dx%d", size, size);
    profiler_start(profiler, label);
    int result = matrix_multiply_naive(A, B, C);
    profiler_end(profiler, label);
    
    if (result != 0) {
        fprintf(stderr, "Matrix multiplication failed\n");
    }
    
    // Cleanup
    snprintf(label, sizeof(label), "matrix_free_%dx%d", size, size);
    profiler_start(profiler, label);
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
    profiler_end(profiler, label);
}

int main(int argc, char* argv[]) {
    Profiler profiler;
    profiler_init(&profiler);
    
    printf("Matrix Multiplication Profiling\n");
    printf("================================\n\n");
    
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
