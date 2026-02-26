# Matrix Multiplication Profiling

A C project demonstrating code execution profiling with multiple matrix multiplication implementations.

## Overview

This project implements and profiles three different matrix multiplication algorithms:
1. **Naive**: Standard triple-nested loop implementation
2. **Transpose-optimized**: Uses transposed B matrix for better cache locality
3. **Cache-blocked**: Uses tiling based on system cache line size for maximum performance

## Project Structure

```
.
├── include/         # Header files
│   ├── matrix.h     # Matrix operations and multiplication functions
│   └── profiler.h   # Profiling utilities
├── src/             # Source files
│   ├── main.c       # Main program and test orchestration
│   ├── matrix.c     # Matrix implementation
│   └── profiler.c   # Profiling implementation
├── CMakeLists.txt   # CMake configuration
├── Makefile         # Legacy Make build (optional)
└── profile.sh       # Profiling script
```

## Building

### Using CMake (Recommended)

```bash
# Create build directory
mkdir build
cd build

# Configure
cmake ..

# Build
cmake --build .

# Or with specific build type
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### Using Make (Legacy)

```bash
# Build the project
make all

# Clean build artifacts
make clean

# Run the profiler
make run

# Run with profiling script
make profile
```

## Running

### Direct execution

```bash
# If built with CMake
./build/bin/matrix_profile

# If built with Make
./bin/matrix_profile
```

### Using the profiling script

```bash
./profile.sh
```

The script will:
- Build the project if needed
- Run the profiler
- Analyze and display results
- Save detailed results to `profile_results.csv`

## Implementation Details

### Matrix Multiplication Algorithms

1. **Naive Implementation**: `matrix_multiply_naive()`
   - Standard i-j-k loop ordering
   - Poor cache locality due to strided access in B matrix

2. **Transpose-optimized**: `matrix_multiply_transpose()`
   - Transposes B matrix first
   - i-j-k ordering with contiguous B access
   - Better cache locality

3. **Cache-blocked**: `matrix_multiply_blocked()`
   - Detects system cache line size
   - Uses tiling to keep data in L1 cache
   - Direct pointer access for maximum performance
   - ~5x speedup over naive for large matrices

### Profiling System

The profiler uses `CLOCK_MONOTONIC` for high-resolution timing:
- Start/end profiling around code sections
- Accumulate time across multiple iterations
- Print formatted results
- Export to CSV for analysis

### Sanity Check

After each multiplication, results are compared across all three methods to ensure correctness. Maximum element-wise difference must be below 1e-9.

## Example Output

```
System cache line size: 64 bytes
L1 data cache size: 49152 bytes (48 KB)
Optimal block size for tiling: 32 x 32

Testing 512x512 matrix multiplication (3 iterations)...

========================================
         PROFILING RESULTS              
========================================
Section                              Time (ms)
----------------------------------------
matrix_multiply_naive_512x512         688.43
matrix_multiply_transpose_512x512     354.73
matrix_multiply_blocked_512x512       138.36
----------------------------------------
TOTAL                                1386.51
========================================
```

## Performance Comparison

For 512x512 matrices:
- **Naive**: ~688 ms
- **Transpose**: ~355 ms (1.9x faster)
- **Cache-blocked**: ~138 ms (5.0x faster)

## Requirements

- GCC or compatible C compiler
- CMake 3.10+
- POSIX-compliant system (for `clock_gettime`)
- Bash (for profiling script)

## License

MIT License