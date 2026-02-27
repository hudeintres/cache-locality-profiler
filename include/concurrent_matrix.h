#ifndef CONCURRENT_MATRIX_H
#define CONCURRENT_MATRIX_H

#include "matrix.h"
#include <vector>
#include <functional>

#ifdef __cplusplus

/**
 * Concurrent matrix multiplication using C++11 threading.
 * 
 * These functions parallelize the three cache-locality optimized
 * matrix multiplication methods to demonstrate speedup from
 * multi-threading combined with cache-efficient algorithms.
 */

/**
 * Concurrent naive matrix multiplication: C = A * B
 * 
 * Parallelizes the naive triple-loop algorithm by distributing rows
 * across threads. Each thread computes a subset of output rows.
 * 
 * @param A First input matrix (M x N)
 * @param B Second input matrix (N x P)
 * @param C Output matrix (M x P), must be pre-allocated
 * @param num_threads Number of threads to use (0 = auto-detect)
 * @return 0 on success, -1 on error
 */
int matrix_multiply_naive_concurrent(Matrix* A, Matrix* B, Matrix* C, int num_threads);

/**
 * Concurrent transpose-optimized matrix multiplication: C = A * B
 * 
 * First transposes B for better cache locality, then parallelizes
 * the computation across threads. Each thread computes a subset of
 * output rows using the transposed B matrix.
 * 
 * @param A First input matrix (M x N)
 * @param B Second input matrix (N x P)
 * @param C Output matrix (M x P), must be pre-allocated
 * @param num_threads Number of threads to use (0 = auto-detect)
 * @return 0 on success, -1 on error
 */
int matrix_multiply_transpose_concurrent(Matrix* A, Matrix* B, Matrix* C, int num_threads);

/**
 * Concurrent cache-blocked matrix multiplication: C = A * B
 * 
 * Parallelizes the blocked/tiling algorithm. Each thread handles
 * a set of block rows, maximizing cache reuse within each thread.
 * 
 * @param A First input matrix (M x N)
 * @param B Second input matrix (N x P)
 * @param C Output matrix (M x P), must be pre-allocated
 * @param block_size Size of cache blocks (0 = auto-detect optimal)
 * @param num_threads Number of threads to use (0 = auto-detect)
 * @return 0 on success, -1 on error
 */
int matrix_multiply_blocked_concurrent(Matrix* A, Matrix* B, Matrix* C, 
                                        int block_size, int num_threads);

/**
 * Get the number of hardware threads available.
 * 
 * @return Number of concurrent threads supported by the hardware
 */
int get_hardware_concurrency();

/**
 * Benchmark result structure for comparing sequential vs concurrent performance.
 */
struct ConcurrentBenchmarkResult {
    const char* method_name;
    double sequential_ms;
    double concurrent_ms;
    double speedup;
    int num_threads;
};

/**
 * Run a comprehensive benchmark comparing all three methods
 * (naive, transpose, blocked) in both sequential and concurrent modes.
 * 
 * @param size Size of square matrices to test
 * @param iterations Number of iterations for averaging
 * @param num_threads Number of threads for concurrent tests (0 = auto)
 * @param results Vector to store benchmark results
 */
void benchmark_concurrent_methods(int size, int iterations, int num_threads,
                                   std::vector<ConcurrentBenchmarkResult>& results);

/**
 * Print benchmark results in a formatted table.
 * 
 * @param results Vector of benchmark results to print
 */
void print_benchmark_results(const std::vector<ConcurrentBenchmarkResult>& results);

/**
 * Save benchmark results to a CSV file.
 * 
 * @param results Vector of benchmark results
 * @param filename Output CSV filename
 */
void save_benchmark_results(const std::vector<ConcurrentBenchmarkResult>& results,
                             const char* filename);

/**
 * Test all concurrent implementations and compare with sequential versions.
 * Runs correctness checks and performance benchmarks.
 * 
 * @param size Matrix size to test
 * @param iterations Number of iterations for timing
 * @param num_threads Number of threads (0 = auto)
 * @param output_file CSV file to save results (or NULL for default)
 */
void test_concurrent_matrix_multiplication(int size, int iterations, 
                                            int num_threads, const char* output_file);

#endif // __cplusplus

#endif // CONCURRENT_MATRIX_H
