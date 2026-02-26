cmake_minimum_required(VERSION 3.10)
project(MatrixProfile C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -D_POSIX_C_SOURCE=199309L")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -D_POSIX_C_SOURCE=199309L")
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/matrix.c
    src/profiler.c
    src/cache_locality.c
)

# Static Library (C)
add_library(matrix_profile_lib STATIC ${SOURCES})
target_link_libraries(matrix_profile_lib m)

# Executable (C)
add_executable(matrix_profile src/main.c)
target_link_libraries(matrix_profile matrix_profile_lib m)

# Static Library (C++)
add_library(matrix_profile_lib_cpp STATIC ${SOURCES})
set_target_properties(matrix_profile_lib_cpp PROPERTIES
    LINKER_LANGUAGE CXX
)
target_link_libraries(matrix_profile_lib_cpp m)

# Executable (C++)
add_executable(matrix_profile_cpp src/main_cpp.cpp)
set_target_properties(matrix_profile_cpp PROPERTIES
    LINKER_LANGUAGE CXX
)
target_link_libraries(matrix_profile_cpp matrix_profile_lib_cpp m)

# Set output directory
set_target_properties(matrix_profile matrix_profile_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
