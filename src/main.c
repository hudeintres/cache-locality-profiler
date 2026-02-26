#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "profiler.h"
#include "matrix.h"
#include "cache_locality.h"

int main(int argc, char* argv[]) {
    // Default sizes
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
        test_cache_locality_speedup(sizes[s], iterations, "profile_results.csv");
    }
    
    if (extra_size > 0) {
        test_cache_locality_speedup(extra_size, iterations, "profile_results.csv");
    }
    
    return 0;
}
