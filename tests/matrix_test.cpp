#include <gtest/gtest.h>
#include "matrix.h"
#include <cmath>

class MatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup
    }

    void TearDown() override {
        // Common teardown
    }
};

TEST_F(MatrixTest, CreateAndFree) {
    Matrix* m = matrix_create(10, 10);
    ASSERT_NE(m, nullptr);
    ASSERT_NE(m->data, nullptr);
    EXPECT_EQ(m->rows, 10);
    EXPECT_EQ(m->cols, 10);
    matrix_free(m);
}

TEST_F(MatrixTest, SetAndGet) {
    Matrix* m = matrix_create(5, 5);
    ASSERT_NE(m, nullptr);
    
    matrix_set(m, 2, 2, 42.0);
    EXPECT_DOUBLE_EQ(matrix_get(m, 2, 2), 42.0);
    
    matrix_set(m, 0, 0, -1.5);
    EXPECT_DOUBLE_EQ(matrix_get(m, 0, 0), -1.5);
    
    // Test out of bounds (should handle gracefully, usually returning 0 or doing nothing)
    matrix_set(m, 10, 10, 100.0);
    EXPECT_DOUBLE_EQ(matrix_get(m, 10, 10), 0.0);
    
    matrix_free(m);
}

TEST_F(MatrixTest, MultiplicationNaive) {
    Matrix* A = matrix_create(2, 2);
    Matrix* B = matrix_create(2, 2);
    Matrix* C = matrix_create(2, 2);
    
    // A = [[1, 2], [3, 4]]
    matrix_set(A, 0, 0, 1.0); matrix_set(A, 0, 1, 2.0);
    matrix_set(A, 1, 0, 3.0); matrix_set(A, 1, 1, 4.0);
    
    // B = [[2, 0], [1, 2]]
    matrix_set(B, 0, 0, 2.0); matrix_set(B, 0, 1, 0.0);
    matrix_set(B, 1, 0, 1.0); matrix_set(B, 1, 1, 2.0);
    
    // Expected C = [[4, 4], [10, 8]]
    // C[0][0] = 1*2 + 2*1 = 4
    // C[0][1] = 1*0 + 2*2 = 4
    // C[1][0] = 3*2 + 4*1 = 10
    // C[1][1] = 3*0 + 4*2 = 8
    
    int result = matrix_multiply_naive(A, B, C);
    EXPECT_EQ(result, 0);
    
    EXPECT_DOUBLE_EQ(matrix_get(C, 0, 0), 4.0);
    EXPECT_DOUBLE_EQ(matrix_get(C, 0, 1), 4.0);
    EXPECT_DOUBLE_EQ(matrix_get(C, 1, 0), 10.0);
    EXPECT_DOUBLE_EQ(matrix_get(C, 1, 1), 8.0);
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

TEST_F(MatrixTest, MultiplicationTranspose) {
    Matrix* A = matrix_create(3, 3);
    Matrix* B = matrix_create(3, 3);
    Matrix* C = matrix_create(3, 3);
    
    // Identity matrix for B
    matrix_zeros(B);
    matrix_set(B, 0, 0, 1.0);
    matrix_set(B, 1, 1, 1.0);
    matrix_set(B, 2, 2, 1.0);
    
    // Random A
    matrix_randomize(A);
    
    int result = matrix_multiply_transpose(A, B, C);
    EXPECT_EQ(result, 0);
    
    // C should equal A
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            EXPECT_DOUBLE_EQ(matrix_get(C, i, j), matrix_get(A, i, j));
        }
    }
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

TEST_F(MatrixTest, MultiplicationBlocked) {
    int size = 16;
    Matrix* A = matrix_create(size, size);
    Matrix* B = matrix_create(size, size);
    Matrix* C_naive = matrix_create(size, size);
    Matrix* C_blocked = matrix_create(size, size);
    
    matrix_randomize(A);
    matrix_randomize(B);
    
    matrix_multiply_naive(A, B, C_naive);
    matrix_multiply_blocked(A, B, C_blocked, 4); // Block size 4
    
    // Compare results
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            EXPECT_NEAR(matrix_get(C_naive, i, j), matrix_get(C_blocked, i, j), 1e-9);
        }
    }
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C_naive);
    matrix_free(C_blocked);
}

TEST_F(MatrixTest, DimensionMismatch) {
    Matrix* A = matrix_create(2, 3);
    Matrix* B = matrix_create(4, 2); // 3 != 4 mismatch
    Matrix* C = matrix_create(2, 2);
    
    EXPECT_EQ(matrix_multiply_naive(A, B, C), -1);
    EXPECT_EQ(matrix_multiply_transpose(A, B, C), -1);
    EXPECT_EQ(matrix_multiply_blocked(A, B, C, 2), -1);
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

// Parallel matrix multiplication tests

TEST_F(MatrixTest, MultiplicationNaiveParallel1Thread) {
    Matrix* A = matrix_create(2, 2);
    Matrix* B = matrix_create(2, 2);
    Matrix* C = matrix_create(2, 2);
    
    // A = [[1, 2], [3, 4]]
    matrix_set(A, 0, 0, 1.0); matrix_set(A, 0, 1, 2.0);
    matrix_set(A, 1, 0, 3.0); matrix_set(A, 1, 1, 4.0);
    
    // B = [[2, 0], [1, 2]]
    matrix_set(B, 0, 0, 2.0); matrix_set(B, 0, 1, 0.0);
    matrix_set(B, 1, 0, 1.0); matrix_set(B, 1, 1, 2.0);
    
    // Expected C = [[4, 4], [10, 8]]
    int result = matrix_multiply_naive_parallel(A, B, C, 1);
    EXPECT_EQ(result, 0);
    
    EXPECT_DOUBLE_EQ(matrix_get(C, 0, 0), 4.0);
    EXPECT_DOUBLE_EQ(matrix_get(C, 0, 1), 4.0);
    EXPECT_DOUBLE_EQ(matrix_get(C, 1, 0), 10.0);
    EXPECT_DOUBLE_EQ(matrix_get(C, 1, 1), 8.0);
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

TEST_F(MatrixTest, MultiplicationNaiveParallel2Threads) {
    Matrix* A = matrix_create(2, 2);
    Matrix* B = matrix_create(2, 2);
    Matrix* C = matrix_create(2, 2);
    
    // A = [[1, 2], [3, 4]]
    matrix_set(A, 0, 0, 1.0); matrix_set(A, 0, 1, 2.0);
    matrix_set(A, 1, 0, 3.0); matrix_set(A, 1, 1, 4.0);
    
    // B = [[2, 0], [1, 2]]
    matrix_set(B, 0, 0, 2.0); matrix_set(B, 0, 1, 0.0);
    matrix_set(B, 1, 0, 1.0); matrix_set(B, 1, 1, 2.0);
    
    // Expected C = [[4, 4], [10, 8]]
    int result = matrix_multiply_naive_parallel(A, B, C, 2);
    EXPECT_EQ(result, 0);
    
    EXPECT_DOUBLE_EQ(matrix_get(C, 0, 0), 4.0);
    EXPECT_DOUBLE_EQ(matrix_get(C, 0, 1), 4.0);
    EXPECT_DOUBLE_EQ(matrix_get(C, 1, 0), 10.0);
    EXPECT_DOUBLE_EQ(matrix_get(C, 1, 1), 8.0);
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

TEST_F(MatrixTest, MultiplicationNaiveParallelMatchesNaive) {
    int size = 32;
    Matrix* A = matrix_create(size, size);
    Matrix* B = matrix_create(size, size);
    Matrix* C_naive = matrix_create(size, size);
    Matrix* C_parallel_t1 = matrix_create(size, size);
    Matrix* C_parallel_t2 = matrix_create(size, size);
    
    matrix_randomize(A);
    matrix_randomize(B);
    
    matrix_multiply_naive(A, B, C_naive);
    matrix_multiply_naive_parallel(A, B, C_parallel_t1, 1);
    matrix_multiply_naive_parallel(A, B, C_parallel_t2, 2);
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            EXPECT_NEAR(matrix_get(C_naive, i, j), matrix_get(C_parallel_t1, i, j), 1e-9);
            EXPECT_NEAR(matrix_get(C_naive, i, j), matrix_get(C_parallel_t2, i, j), 1e-9);
        }
    }
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C_naive);
    matrix_free(C_parallel_t1);
    matrix_free(C_parallel_t2);
}

TEST_F(MatrixTest, MultiplicationTransposeParallel1Thread) {
    Matrix* A = matrix_create(3, 3);
    Matrix* B = matrix_create(3, 3);
    Matrix* C = matrix_create(3, 3);
    
    // Identity matrix for B
    matrix_zeros(B);
    matrix_set(B, 0, 0, 1.0);
    matrix_set(B, 1, 1, 1.0);
    matrix_set(B, 2, 2, 1.0);
    
    matrix_randomize(A);
    
    int result = matrix_multiply_transpose_parallel(A, B, C, 1);
    EXPECT_EQ(result, 0);
    
    // C should equal A
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_DOUBLE_EQ(matrix_get(C, i, j), matrix_get(A, i, j));
        }
    }
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

TEST_F(MatrixTest, MultiplicationTransposeParallel2Threads) {
    Matrix* A = matrix_create(3, 3);
    Matrix* B = matrix_create(3, 3);
    Matrix* C = matrix_create(3, 3);
    
    // Identity matrix for B
    matrix_zeros(B);
    matrix_set(B, 0, 0, 1.0);
    matrix_set(B, 1, 1, 1.0);
    matrix_set(B, 2, 2, 1.0);
    
    matrix_randomize(A);
    
    int result = matrix_multiply_transpose_parallel(A, B, C, 2);
    EXPECT_EQ(result, 0);
    
    // C should equal A
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_DOUBLE_EQ(matrix_get(C, i, j), matrix_get(A, i, j));
        }
    }
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

TEST_F(MatrixTest, MultiplicationTransposeParallelMatchesTranspose) {
    int size = 32;
    Matrix* A = matrix_create(size, size);
    Matrix* B = matrix_create(size, size);
    Matrix* C_transpose = matrix_create(size, size);
    Matrix* C_parallel_t1 = matrix_create(size, size);
    Matrix* C_parallel_t2 = matrix_create(size, size);
    
    matrix_randomize(A);
    matrix_randomize(B);
    
    matrix_multiply_transpose(A, B, C_transpose);
    matrix_multiply_transpose_parallel(A, B, C_parallel_t1, 1);
    matrix_multiply_transpose_parallel(A, B, C_parallel_t2, 2);
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            EXPECT_NEAR(matrix_get(C_transpose, i, j), matrix_get(C_parallel_t1, i, j), 1e-9);
            EXPECT_NEAR(matrix_get(C_transpose, i, j), matrix_get(C_parallel_t2, i, j), 1e-9);
        }
    }
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C_transpose);
    matrix_free(C_parallel_t1);
    matrix_free(C_parallel_t2);
}

TEST_F(MatrixTest, MultiplicationBlockedParallel1Thread) {
    int size = 16;
    Matrix* A = matrix_create(size, size);
    Matrix* B = matrix_create(size, size);
    Matrix* C_naive = matrix_create(size, size);
    Matrix* C_blocked_parallel = matrix_create(size, size);
    
    matrix_randomize(A);
    matrix_randomize(B);
    
    matrix_multiply_naive(A, B, C_naive);
    int result = matrix_multiply_blocked_parallel(A, B, C_blocked_parallel, 4, 1);
    EXPECT_EQ(result, 0);
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            EXPECT_NEAR(matrix_get(C_naive, i, j), matrix_get(C_blocked_parallel, i, j), 1e-9);
        }
    }
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C_naive);
    matrix_free(C_blocked_parallel);
}

TEST_F(MatrixTest, MultiplicationBlockedParallel2Threads) {
    int size = 16;
    Matrix* A = matrix_create(size, size);
    Matrix* B = matrix_create(size, size);
    Matrix* C_naive = matrix_create(size, size);
    Matrix* C_blocked_parallel = matrix_create(size, size);
    
    matrix_randomize(A);
    matrix_randomize(B);
    
    matrix_multiply_naive(A, B, C_naive);
    int result = matrix_multiply_blocked_parallel(A, B, C_blocked_parallel, 4, 2);
    EXPECT_EQ(result, 0);
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            EXPECT_NEAR(matrix_get(C_naive, i, j), matrix_get(C_blocked_parallel, i, j), 1e-9);
        }
    }
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C_naive);
    matrix_free(C_blocked_parallel);
}

TEST_F(MatrixTest, MultiplicationBlockedParallelMatchesBlocked) {
    int size = 32;
    Matrix* A = matrix_create(size, size);
    Matrix* B = matrix_create(size, size);
    Matrix* C_blocked = matrix_create(size, size);
    Matrix* C_parallel_t1 = matrix_create(size, size);
    Matrix* C_parallel_t2 = matrix_create(size, size);
    
    matrix_randomize(A);
    matrix_randomize(B);
    
    matrix_multiply_blocked(A, B, C_blocked, 8);
    matrix_multiply_blocked_parallel(A, B, C_parallel_t1, 8, 1);
    matrix_multiply_blocked_parallel(A, B, C_parallel_t2, 8, 2);
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            EXPECT_NEAR(matrix_get(C_blocked, i, j), matrix_get(C_parallel_t1, i, j), 1e-9);
            EXPECT_NEAR(matrix_get(C_blocked, i, j), matrix_get(C_parallel_t2, i, j), 1e-9);
        }
    }
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C_blocked);
    matrix_free(C_parallel_t1);
    matrix_free(C_parallel_t2);
}

TEST_F(MatrixTest, ParallelDimensionMismatch) {
    Matrix* A = matrix_create(2, 3);
    Matrix* B = matrix_create(4, 2); // 3 != 4 mismatch
    Matrix* C = matrix_create(2, 2);
    
    EXPECT_EQ(matrix_multiply_naive_parallel(A, B, C, 1), -1);
    EXPECT_EQ(matrix_multiply_naive_parallel(A, B, C, 2), -1);
    EXPECT_EQ(matrix_multiply_transpose_parallel(A, B, C, 1), -1);
    EXPECT_EQ(matrix_multiply_transpose_parallel(A, B, C, 2), -1);
    EXPECT_EQ(matrix_multiply_blocked_parallel(A, B, C, 2, 1), -1);
    EXPECT_EQ(matrix_multiply_blocked_parallel(A, B, C, 2, 2), -1);
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

TEST_F(MatrixTest, ParallelNullMatrixHandling) {
    Matrix* A = matrix_create(2, 2);
    Matrix* B = matrix_create(2, 2);
    Matrix* C = matrix_create(2, 2);
    
    matrix_randomize(A);
    matrix_randomize(B);
    
    // Test null handling
    EXPECT_EQ(matrix_multiply_naive_parallel(nullptr, B, C, 1), -1);
    EXPECT_EQ(matrix_multiply_naive_parallel(A, nullptr, C, 1), -1);
    EXPECT_EQ(matrix_multiply_naive_parallel(A, B, nullptr, 1), -1);
    EXPECT_EQ(matrix_multiply_transpose_parallel(nullptr, B, C, 1), -1);
    EXPECT_EQ(matrix_multiply_blocked_parallel(nullptr, B, C, 4, 1), -1);
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

TEST_F(MatrixTest, ParallelLargeMatrix) {
    int size = 128;
    Matrix* A = matrix_create(size, size);
    Matrix* B = matrix_create(size, size);
    Matrix* C_naive = matrix_create(size, size);
    Matrix* C_parallel_t1 = matrix_create(size, size);
    Matrix* C_parallel_t2 = matrix_create(size, size);
    
    matrix_randomize(A);
    matrix_randomize(B);
    
    matrix_multiply_naive(A, B, C_naive);
    matrix_multiply_naive_parallel(A, B, C_parallel_t1, 1);
    matrix_multiply_naive_parallel(A, B, C_parallel_t2, 2);
    
    // Verify all results match
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            EXPECT_NEAR(matrix_get(C_naive, i, j), matrix_get(C_parallel_t1, i, j), 1e-9);
            EXPECT_NEAR(matrix_get(C_naive, i, j), matrix_get(C_parallel_t2, i, j), 1e-9);
        }
    }
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C_naive);
    matrix_free(C_parallel_t1);
    matrix_free(C_parallel_t2);
}
