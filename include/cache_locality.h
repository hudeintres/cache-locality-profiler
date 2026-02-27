#ifndef CACHE_LOCALITY_H
#define CACHE_LOCALITY_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run the matrix multiplication cache locality test.
 *
 * @param size The size of the square matrices to multiply (e.g., 512 for 512x512)
 * @param iterations The number of iterations to run for averaging results
 * @param output_file The path to save the CSV results (or NULL for default "profile_results.csv")
 */
void test_cache_locality_speedup(int size, int iterations, const char* output_file);

#ifdef __cplusplus
}
#endif

#endif // CACHE_LOCALITY_H
