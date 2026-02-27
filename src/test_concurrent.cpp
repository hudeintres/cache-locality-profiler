#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include "concurrent_matrix.h"
#include "matrix.h"

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --size <N>       Matrix size (default: 512)\n";
    std::cout << "  --threads <N>    Number of threads (0 = auto, default: auto)\n";
    std::cout << "  --iterations <N> Number of iterations (default: 3)\n";
    std::cout << "  --output <file>  Output CSV file (default: concurrent_benchmark.csv)\n";
    std::cout << "  --help           Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " --size 1024 --threads 4\n";
    std::cout << "  " << program_name << " --size 2048 --iterations 5\n";
}

int main(int argc, char* argv[]) {
    // Default values
    int size = 512;
    int num_threads = 0;  // 0 = auto-detect
    int iterations = 3;
    const char* output_file = "concurrent_benchmark.csv";
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            size = std::atoi(argv[++i]);
            if (size <= 0 || size > 4096) {
                std::cerr << "Error: Size must be between 1 and 4096\n";
                return 1;
            }
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
            if (num_threads < 0) {
                std::cerr << "Error: Threads must be >= 0\n";
                return 1;
            }
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
            if (iterations <= 0) {
                std::cerr << "Error: Iterations must be > 0\n";
                return 1;
            }
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else {
            // Check if it's just a number (positional size argument)
            std::string arg = argv[i];
            bool is_number = true;
            for (char c : arg) {
                if (!std::isdigit(c)) {
                    is_number = false;
                    break;
                }
            }
            if (is_number && !arg.empty()) {
                size = std::atoi(argv[i]);
                if (size <= 0 || size > 4096) {
                    std::cerr << "Error: Size must be between 1 and 4096\n";
                    return 1;
                }
            } else {
                std::cerr << "Unknown option: " << argv[i] << "\n";
                print_usage(argv[0]);
                return 1;
            }
        }
    }
    
    // Run the concurrent matrix multiplication test
    test_concurrent_matrix_multiplication(size, iterations, num_threads, output_file);
    
    return 0;
}
