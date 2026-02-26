#!/bin/bash

# Profiling script for matrix multiplication
# Runs the profiler, analyzes results, and generates a report

set -e

echo "=============================================="
echo "     Matrix Multiplication Profiling Tool    "
echo "=============================================="
echo ""

# Check if binary exists
if [ ! -f "build/matrix_profile" ]; then
    echo "Building project..."
    mkdir -p build
    cd build
    cmake ..
    make
    cd ..
    echo ""
fi

# Run the profiler
echo "Running profiler..."
echo "----------------------------------------------"
./build/matrix_profile

# Analyze results if CSV file was created
if [ -f "profile_results.csv" ]; then
    echo ""
    echo "=============================================="
    echo "           Analysis Report                    "
    echo "=============================================="
    echo ""
    
    # Parse and display results nicely
    echo "Detailed Results:"
    echo "----------------------------------------------"
    
    # Skip header and sort by time (descending)
    tail -n +2 profile_results.csv | sort -t',' -k2 -nr | head -20 | while IFS=',' read -r section time_ms; do
        printf "%-40s %10.4f ms\n" "$section" "$time_ms"
    done
    
    echo ""
    echo "Matrix Multiplication Only (sorted by size):"
    echo "----------------------------------------------"
    
    # Extract multiplication times (naive vs transpose vs blocked)
    echo "Naive multiplication:"
    grep "matrix_multiply_naive" profile_results.csv | while IFS=',' read -r section time_ms; do
        # Extract size from section name
        size=$(echo "$section" | grep -oP '\d+x\d+' | head -1)
        printf "Size %s: %10.4f ms\n" "$size" "$time_ms"
    done
    
    echo ""
    echo "Transpose-optimized multiplication:"
    grep "matrix_multiply_transpose" profile_results.csv | while IFS=',' read -r section time_ms; do
        # Extract size from section name
        size=$(echo "$section" | grep -oP '\d+x\d+' | head -1)
        printf "Size %s: %10.4f ms\n" "$size" "$time_ms"
    done
    
    echo ""
    echo "Cache-blocked multiplication (tiling):"
    grep "matrix_multiply_blocked" profile_results.csv | while IFS=',' read -r section time_ms; do
        # Extract size from section name
        size=$(echo "$section" | grep -oP '\d+x\d+' | head -1)
        printf "Size %s: %10.4f ms\n" "$size" "$time_ms"
    done
    
    echo ""
    echo "Full results saved to: profile_results.csv"
fi

echo ""
echo "Profiling complete!"
