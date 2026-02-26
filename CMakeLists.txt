cmake_minimum_required(VERSION 3.10)
project(MatrixProfile C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -D_POSIX_C_SOURCE=199309L")
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/main.c
    src/matrix.c
    src/profiler.c
)

# Executable
add_executable(matrix_profile ${SOURCES})

# Link math library
target_link_libraries(matrix_profile m)

# Set output directory
set_target_properties(matrix_profile PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
