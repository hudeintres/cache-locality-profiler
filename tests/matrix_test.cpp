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

// ============================================================================
// Concurrent Matrix Multiplication Tests
// ============================================================================

#include "concurrent_matrix.h"

TEST_F(MatrixTest, ConcurrentNaiveMultiplication) {
    Matrix* A = matrix_create(4, 4);
    Matrix* B = matrix_create(4, 4);
    Matrix* C_seq = matrix_create(4, 4);
    Matrix* C_conc = matrix_create(4, 4);
    
    // A = [[1, 2], [3, 4]] extended to 4x4
    matrix_set(A, 0, 0, 1.0); matrix_set(A, 0, 1, 2.0); matrix_set(A, 0, 2, 3.0); matrix_set(A, 0, 3, 4.0);
    matrix_set(A, 1, 0, 5.0); matrix_set(A, 1, 1, 6.0); matrix_set(A, 1, 2, 7.0); matrix_set(A, 1, 3, 8.0);
    matrix_set(A, 2, 0, 9.0); matrix_set(A, 2, 1, 10.0); matrix_set(A, 2, 2, 11.0); matrix_set(A, 2, 3, 12.0);
    matrix_set(A, 3, 0, 13.0); matrix_set(A, 3, 1, 14.0); matrix_set(A, 3, 2, 15.0); matrix_set(A, 3, 3, 16.0);
    
    // B = identity
    matrix_zeros(B);
    matrix_set(B, 0, 0, 1.0);
    matrix_set(B, 1, 1, 1.0);
    matrix_set(B, 2, 2, 1.0);
    matrix_set(B, 3, 3, 1.0);
    
    int result_seq = matrix_multiply_naive(A, B, C_seq);
    int result_conc = matrix_multiply_naive_concurrent(A, B, C_conc, 2);
    
    EXPECT_EQ(result_seq, 0);
    EXPECT_EQ(result_conc, 0);
    
    // C should equal A (since B is identity)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            EXPECT_NEAR(matrix_get(C_seq, i, j), matrix_get(A, i, j), 1e-9);
            EXPECT_NEAR(matrix_get(C_conc, i, j), matrix_get(A, i, j), 1e-9);
        }
    }
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C_seq);
    matrix_free(C_conc);
}

TEST_F(MatrixTest, ConcurrentTransposeMultiplication) {
    int size = 32;
    Matrix* A = matrix_create(size, size);
    Matrix* B = matrix_create(size, size);
    Matrix* C_seq = matrix_create(size, size);
    Matrix* C_conc = matrix_create(size, size);
    
    matrix_randomize(A);
    matrix_randomize(B);
    
    int result_seq = matrix_multiply_transpose(A, B, C_seq);
    int result_conc = matrix_multiply_transpose_concurrent(A, B, C_conc, 4);
    
    EXPECT_EQ(result_seq, 0);
    EXPECT_EQ(result_conc, 0);
    
    // Results should match
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            EXPECT_NEAR(matrix_get(C_seq, i, j), matrix_get(C_conc, i, j), 1e-9);
        }
    }
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C_seq);
    matrix_free(C_conc);
}

TEST_F(MatrixTest, ConcurrentBlockedMultiplication) {
    int size = 64;
    Matrix* A = matrix_create(size, size);
    Matrix* B = matrix_create(size, size);
    Matrix* C_seq = matrix_create(size, size);
    Matrix* C_conc = matrix_create(size, size);
    
    matrix_randomize(A);
    matrix_randomize(B);
    
    int result_seq = matrix_multiply_blocked(A, B, C_seq, 16);
    int result_conc = matrix_multiply_blocked_concurrent(A, B, C_conc, 16, 4);
    
    EXPECT_EQ(result_seq, 0);
    EXPECT_EQ(result_conc, 0);
    
    // Results should match
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            EXPECT_NEAR(matrix_get(C_seq, i, j), matrix_get(C_conc, i, j), 1e-9);
        }
    }
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C_seq);
    matrix_free(C_conc);
}

TEST_F(MatrixTest, ConcurrentMethodsConsistency) {
    int size = 32;
    Matrix* A = matrix_create(size, size);
    Matrix* B = matrix_create(size, size);
    Matrix* C_naive = matrix_create(size, size);
    Matrix* C_transpose = matrix_create(size, size);
    Matrix* C_blocked = matrix_create(size, size);
    
    matrix_randomize(A);
    matrix_randomize(B);
    
    // All concurrent methods should produce same result
    matrix_multiply_naive_concurrent(A, B, C_naive, 2);
    matrix_multiply_transpose_concurrent(A, B, C_transpose, 2);
    matrix_multiply_blocked_concurrent(A, B, C_blocked, 0, 2);  // auto block size
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            EXPECT_NEAR(matrix_get(C_naive, i, j), matrix_get(C_transpose, i, j), 1e-9);
            EXPECT_NEAR(matrix_get(C_naive, i, j), matrix_get(C_blocked, i, j), 1e-9);
        }
    }
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C_naive);
    matrix_free(C_transpose);
    matrix_free(C_blocked);
}

TEST_F(MatrixTest, ConcurrentDimensionMismatch) {
    Matrix* A = matrix_create(2, 3);
    Matrix* B = matrix_create(4, 2);
    Matrix* C = matrix_create(2, 2);
    
    EXPECT_EQ(matrix_multiply_naive_concurrent(A, B, C, 2), -1);
    EXPECT_EQ(matrix_multiply_transpose_concurrent(A, B, C, 2), -1);
    EXPECT_EQ(matrix_multiply_blocked_concurrent(A, B, C, 4, 2), -1);
    
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

TEST_F(MatrixTest, HardwareConcurrency) {
    int threads = get_hardware_concurrency();
    EXPECT_GT(threads, 0);
    EXPECT_LE(threads, 1024);  // Reasonable upper bound
}
