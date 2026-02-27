cmake_minimum_required(VERSION 3.10)
project(MatrixProfile CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -D_POSIX_C_SOURCE=199309L")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/matrix.c
    src/profiler.c
    src/cache_locality.c
    src/matrix_parallel.cpp
)

set_source_files_properties(
    src/matrix.c
    src/profiler.c
    src/cache_locality.c
    PROPERTIES LANGUAGE CXX
)

# Static Library (C++)
add_library(matrix_profile_lib_cpp STATIC ${SOURCES})
set_target_properties(matrix_profile_lib_cpp PROPERTIES
    LINKER_LANGUAGE CXX
)
target_link_libraries(matrix_profile_lib_cpp m pthread)

# Dynamic Library (C++)
add_library(matrix_profile_lib_cpp_shared SHARED ${SOURCES})
set_target_properties(matrix_profile_lib_cpp_shared PROPERTIES
    LINKER_LANGUAGE CXX
    OUTPUT_NAME matrix_profile
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(matrix_profile_lib_cpp_shared m pthread)

# Executable (C++) - uses static library
add_executable(matrix_profile_cpp src/main_cpp.cpp)
set_target_properties(matrix_profile_cpp PROPERTIES
    LINKER_LANGUAGE CXX
)
target_link_libraries(matrix_profile_cpp matrix_profile_lib_cpp m pthread)

# Executable (C++) - uses dynamic library
add_executable(matrix_profile_cpp_shared src/main_cpp.cpp)
set_target_properties(matrix_profile_cpp_shared PROPERTIES
    LINKER_LANGUAGE CXX
)
target_link_libraries(matrix_profile_cpp_shared matrix_profile_lib_cpp_shared m pthread)

# Set output directory
set_target_properties(matrix_profile_cpp matrix_profile_cpp_shared PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
set_target_properties(matrix_profile_lib_cpp_shared PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Testing
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(matrix_test tests/matrix_test.cpp)
set_target_properties(matrix_test PROPERTIES
    LINKER_LANGUAGE CXX
)
target_link_libraries(matrix_test matrix_profile_lib_cpp_shared GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(matrix_test)
