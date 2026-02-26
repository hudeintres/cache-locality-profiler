#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include "cache_locality.h"

int main(int argc, char* argv[]) {
    std::cout << "Matrix Multiplication Profiling (C++ Interface)" << std::endl;
    std::cout << "=============================================" << std::endl << std::endl;

    // Default sizes
    std::vector<int> sizes = {64, 128, 256, 512};
    
    // Optional size from command line
    int extra_size = 0;
    if (argc > 1) {
        try {
            std::string arg = argv[1];
            size_t pos;
            long value = std::stol(arg, &pos);
            
            if (pos != arg.length() || value <= 0) {
                std::cerr << "Invalid size '" << argv[1] << "'. Using default sizes only." << std::endl;
            } else if (value > 4096) {
                std::cerr << "Requested size " << value << " too large (max 4096). Using default sizes only." << std::endl;
            } else {
                extra_size = static_cast<int>(value);
                std::cout << "Adding user-specified size: " << extra_size << "x" << extra_size << std::endl;
            }
        } catch (...) {
            std::cerr << "Invalid size '" << argv[1] << "'. Using default sizes only." << std::endl;
        }
    }
    
    // Number of iterations for averaging
    int iterations = 3;
    
    for (int size : sizes) {
        test_cache_locality_speedup(size, iterations, "profile_results_cpp.csv");
    }
    
    if (extra_size > 0) {
        test_cache_locality_speedup(extra_size, iterations, "profile_results_cpp.csv");
    }
    
    return 0;
}
