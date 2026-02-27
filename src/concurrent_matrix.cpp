#include "concurrent_matrix.h"
#include "profiler.h"
#include <thread>
#include <vector>
#include <mutex>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>

// Get the number of hardware threads available
int get_hardware_concurrency() {
    unsigned int n = std::thread::hardware_concurrency();
    return (n > 0) ? static_cast<int>(n) : 1;
}

// ============================================================================
// Concurrent Naive Matrix Multiplication
// ============================================================================

// Worker function for naive multiplication - processes a range of rows
static void naive_multiply_worker(Matrix* A, Matrix* B, Matrix* C, 
                                   int start_row, int end_row) {
    int N = A->cols;
    int P = B->cols;
    int a_stride = A->cols;
    int b_stride = B->cols;
    int c_stride = C->cols;
    
    double* a_data = A->data;
    double* b_data = B->data;
    double* c_data = C->data;
    
    // Process rows [start_row, end_row)
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < P; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += a_data[i * a_stride + k] * b_data[k * b_stride + j];
            }
            c_data[i * c_stride + j] = sum;
        }
    }
}

int matrix_multiply_naive_concurrent(Matrix* A, Matrix* B, Matrix* C, int num_threads) {
    // Validate inputs
    if (!A || !B || !C) return -1;
    if (A->cols != B->rows) return -1;
    if (C->rows != A->rows || C->cols != B->cols) return -1;
    
    int M = A->rows;
    
    // Determine number of threads
    int actual_threads = (num_threads > 0) ? num_threads : get_hardware_concurrency();
    actual_threads = std::min(actual_threads, M);  // Can't have more threads than rows
    
    if (actual_threads <= 1) {
        // Fall back to sequential for small matrices or single thread
        return matrix_multiply_naive(A, B, C);
    }
    
    // Create threads
    std::vector<std::thread> threads;
    int rows_per_thread = (M + actual_threads - 1) / actual_threads;
    
    for (int t = 0; t < actual_threads; t++) {
        int start_row = t * rows_per_thread;
        int end_row = std::min((t + 1) * rows_per_thread, M);
        
        if (start_row < end_row) {
            threads.emplace_back(naive_multiply_worker, A, B, C, start_row, end_row);
        }
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    return 0;
}

// ============================================================================
// Concurrent Transpose-Optimized Matrix Multiplication
// ============================================================================

// Worker function for transpose-optimized multiplication
static void transpose_multiply_worker(Matrix* A, Matrix* B_T, Matrix* C,
                                       int start_row, int end_row) {
    int N = A->cols;
    int P = B_T->rows;  // B_T is P x N
    int a_stride = A->cols;
    int bt_stride = B_T->cols;  // N
    int c_stride = C->cols;
    
    double* a_data = A->data;
    double* bt_data = B_T->data;
    double* c_data = C->data;
    
    // Process rows [start_row, end_row)
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < P; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                // A[i][k] and B_T[j][k] both have contiguous access
                sum += a_data[i * a_stride + k] * bt_data[j * bt_stride + k];
            }
            c_data[i * c_stride + j] = sum;
        }
    }
}

int matrix_multiply_transpose_concurrent(Matrix* A, Matrix* B, Matrix* C, int num_threads) {
    // Validate inputs
    if (!A || !B || !C) return -1;
    if (A->cols != B->rows) return -1;
    if (C->rows != A->rows || C->cols != B->cols) return -1;
    
    int M = A->rows;
    int N = A->cols;
    int P = B->cols;
    
    // Determine number of threads
    int actual_threads = (num_threads > 0) ? num_threads : get_hardware_concurrency();
    actual_threads = std::min(actual_threads, M);
    
    if (actual_threads <= 1) {
        return matrix_multiply_transpose(A, B, C);
    }
    
    // Create transposed matrix of B (B_T is P x N)
    Matrix* B_T = matrix_create(P, N);
    if (!B_T) return -1;
    
    // Transpose B into B_T (can also be parallelized for large matrices)
    double* b_data = B->data;
    double* bt_data = B_T->data;
    int b_stride = B->cols;
    
    for (int i = 0; i < B->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            bt_data[j * N + i] = b_data[i * b_stride + j];
        }
    }
    
    // Create threads for multiplication
    std::vector<std::thread> threads;
    int rows_per_thread = (M + actual_threads - 1) / actual_threads;
    
    for (int t = 0; t < actual_threads; t++) {
        int start_row = t * rows_per_thread;
        int end_row = std::min((t + 1) * rows_per_thread, M);
        
        if (start_row < end_row) {
            threads.emplace_back(transpose_multiply_worker, A, B_T, C, start_row, end_row);
        }
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    matrix_free(B_T);
    return 0;
}

// ============================================================================
// Concurrent Cache-Blocked Matrix Multiplication
// ============================================================================

// Worker function for blocked multiplication - processes a range of block rows
static void blocked_multiply_worker(Matrix* A, Matrix* B, Matrix* C,
                                     int M, int N, int P, int BLOCK,
                                     int start_ii, int end_ii) {
    double* a_data = A->data;
    double* b_data = B->data;
    double* c_data = C->data;
    int a_stride = A->cols;
    int b_stride = B->cols;
    int c_stride = C->cols;
    
    // Process block rows [start_ii, end_ii)
    for (int ii = start_ii; ii < end_ii; ii += BLOCK) {
        int i_max = (ii + BLOCK < M) ? ii + BLOCK : M;
        
        for (int kk = 0; kk < N; kk += BLOCK) {
            int k_max = (kk + BLOCK < N) ? kk + BLOCK : N;
            
            for (int jj = 0; jj < P; jj += BLOCK) {
                int j_max = (jj + BLOCK < P) ? jj + BLOCK : P;
                
                // Multiply current blocks
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        double a_val = a_data[i * a_stride + k];
                        for (int j = jj; j < j_max; j++) {
                            c_data[i * c_stride + j] += a_val * b_data[k * b_stride + j];
                        }
                    }
                }
            }
        }
    }
}

int matrix_multiply_blocked_concurrent(Matrix* A, Matrix* B, Matrix* C, 
                                        int block_size, int num_threads) {
    // Validate inputs
    if (!A || !B || !C) return -1;
    if (A->cols != B->rows) return -1;
    if (C->rows != A->rows || C->cols != B->cols) return -1;
    
    int M = A->rows;
    int N = A->cols;
    int P = B->cols;
    
    // Calculate optimal block size if not provided
    int BLOCK;
    if (block_size <= 0) {
        int l1_size = get_l1_cache_size();
        int max_elements = l1_size / (4 * sizeof(double));
        BLOCK = 1;
        while (BLOCK * BLOCK <= max_elements && BLOCK < 128) {
            BLOCK *= 2;
        }
        BLOCK /= 2;
        if (BLOCK < 16) BLOCK = 16;
    } else {
        BLOCK = block_size;
    }
    
    // Initialize C to zeros
    matrix_zeros(C);
    
    // Determine number of threads
    int actual_threads = (num_threads > 0) ? num_threads : get_hardware_concurrency();
    actual_threads = std::min(actual_threads, (M + BLOCK - 1) / BLOCK);
    
    if (actual_threads <= 1) {
        return matrix_multiply_blocked(A, B, C, block_size);
    }
    
    // Create threads - distribute block rows
    std::vector<std::thread> threads;
    int block_rows_per_thread = ((M + BLOCK - 1) / BLOCK + actual_threads - 1) / actual_threads;
    int block_row_step = block_rows_per_thread * BLOCK;
    
    for (int t = 0; t < actual_threads; t++) {
        int start_ii = t * block_row_step;
        int end_ii = std::min((t + 1) * block_row_step, M);
        
        if (start_ii < end_ii) {
            threads.emplace_back(blocked_multiply_worker, A, B, C, 
                                  M, N, P, BLOCK, start_ii, end_ii);
        }
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    return 0;
}

// ============================================================================
// Benchmarking Functions
// ============================================================================

// Helper to get current time in milliseconds
static double get_time_ms_internal() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::milli>(duration).count();
}

void benchmark_concurrent_methods(int size, int iterations, int num_threads,
                                   std::vector<ConcurrentBenchmarkResult>& results) {
    results.clear();
    
    int actual_threads = (num_threads > 0) ? num_threads : get_hardware_concurrency();
    
    // Create matrices
    Matrix* A = matrix_create(size, size);
    Matrix* B = matrix_create(size, size);
    Matrix* C_seq = matrix_create(size, size);
    Matrix* C_conc = matrix_create(size, size);
    
    if (!A || !B || !C_seq || !C_conc) {
        std::cerr << "Failed to allocate matrices for benchmarking" << std::endl;
        if (A) matrix_free(A);
        if (B) matrix_free(B);
        if (C_seq) matrix_free(C_seq);
        if (C_conc) matrix_free(C_conc);
        return;
    }
    
    matrix_randomize(A);
    matrix_randomize(B);
    
    // Calculate optimal block size
    int l1_size = get_l1_cache_size();
    int max_elements = l1_size / (4 * sizeof(double));
    int block_size = 1;
    while (block_size * block_size <= max_elements && block_size < 128) {
        block_size *= 2;
    }
    block_size /= 2;
    if (block_size < 16) block_size = 16;
    
    // Benchmark each method
    const char* method_names[] = {"Naive", "Transpose", "Blocked"};
    
    for (int method = 0; method < 3; method++) {
        ConcurrentBenchmarkResult result;
        result.method_name = method_names[method];
        result.num_threads = actual_threads;
        result.sequential_ms = 0.0;
        result.concurrent_ms = 0.0;
        
        // Sequential benchmark
        for (int iter = 0; iter < iterations; iter++) {
            matrix_zeros(C_seq);
            
            double start_time = get_time_ms_internal();
            
            switch (method) {
                case 0: matrix_multiply_naive(A, B, C_seq); break;
                case 1: matrix_multiply_transpose(A, B, C_seq); break;
                case 2: matrix_multiply_blocked(A, B, C_seq, block_size); break;
            }
            
            double end_time = get_time_ms_internal();
            result.sequential_ms += (end_time - start_time);
        }
        result.sequential_ms /= iterations;
        
        // Concurrent benchmark
        for (int iter = 0; iter < iterations; iter++) {
            matrix_zeros(C_conc);
            
            double start_time = get_time_ms_internal();
            
            switch (method) {
                case 0: matrix_multiply_naive_concurrent(A, B, C_conc, actual_threads); break;
                case 1: matrix_multiply_transpose_concurrent(A, B, C_conc, actual_threads); break;
                case 2: matrix_multiply_blocked_concurrent(A, B, C_conc, block_size, actual_threads); break;
            }
            
            double end_time = get_time_ms_internal();
            result.concurrent_ms += (end_time - start_time);
        }
        result.concurrent_ms /= iterations;
        
        // Calculate speedup
        result.speedup = result.sequential_ms / result.concurrent_ms;
        
        // Verify correctness
        double max_diff = 0.0;
        for (int i = 0; i < size * size; i++) {
            double diff = std::abs(C_seq->data[i] - C_conc->data[i]);
            if (diff > max_diff) max_diff = diff;
        }
        
        if (max_diff > 1e-9) {
            std::cerr << "Warning: " << method_names[method] 
                      << " concurrent result differs from sequential (max diff: " 
                      << max_diff << ")" << std::endl;
        }
        
        results.push_back(result);
    }
    
    // Cleanup
    matrix_free(A);
    matrix_free(B);
    matrix_free(C_seq);
    matrix_free(C_conc);
}

void print_benchmark_results(const std::vector<ConcurrentBenchmarkResult>& results) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Concurrent Matrix Multiplication Benchmark Results            ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Method      │ Sequential (ms) │ Concurrent (ms) │ Speedup │ Threads ║\n";
    std::cout << "╠═════════════╪═════════════════╪═════════════════╪═════════╪═════════╣\n";
    
    for (const auto& result : results) {
        std::cout << "║ " << std::left << std::setw(11) << result.method_name
                  << " │ " << std::right << std::setw(15) << std::fixed << std::setprecision(2) << result.sequential_ms
                  << " │ " << std::setw(15) << result.concurrent_ms
                  << " │ " << std::setw(7) << std::setprecision(2) << result.speedup << "x"
                  << " │ " << std::setw(7) << result.num_threads << " ║\n";
    }
    
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";
    std::cout << std::endl;
}

void save_benchmark_results(const std::vector<ConcurrentBenchmarkResult>& results,
                             const char* filename) {
    std::ofstream file(filename, std::ios::app);  // Append mode
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Check if file is empty, write header if so
    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "method,sequential_ms,concurrent_ms,speedup,num_threads\n";
    }
    
    for (const auto& result : results) {
        file << result.method_name << ","
             << std::fixed << std::setprecision(3) << result.sequential_ms << ","
             << result.concurrent_ms << ","
             << result.speedup << ","
             << result.num_threads << "\n";
    }
    
    file.close();
}

void test_concurrent_matrix_multiplication(int size, int iterations, 
                                            int num_threads, const char* output_file) {
    std::cout << "\n";
    std::cout << "========================================================\n";
    std::cout << "  Concurrent Matrix Multiplication Performance Test\n";
    std::cout << "========================================================\n\n";
    
    int actual_threads = (num_threads > 0) ? num_threads : get_hardware_concurrency();
    
    std::cout << "Matrix size: " << size << " x " << size << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Threads: " << actual_threads << "\n";
    std::cout << "Hardware concurrency: " << get_hardware_concurrency() << "\n";
    
    // Get cache info
    int cache_line_size = get_cache_line_size();
    int l1_cache_size = get_l1_cache_size();
    
    std::cout << "\nCache Information:\n";
    std::cout << "  Cache line size: " << cache_line_size << " bytes\n";
    std::cout << "  L1 data cache: " << l1_cache_size << " bytes (" 
              << (l1_cache_size / 1024) << " KB)\n";
    
    // Run benchmarks
    std::vector<ConcurrentBenchmarkResult> results;
    benchmark_concurrent_methods(size, iterations, actual_threads, results);
    
    // Print results
    print_benchmark_results(results);
    
    // Save to file
    const char* file_to_save = output_file ? output_file : "concurrent_benchmark.csv";
    save_benchmark_results(results, file_to_save);
    std::cout << "Results saved to: " << file_to_save << "\n";
}
