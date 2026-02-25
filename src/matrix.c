#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

Matrix* matrix_create(int rows, int cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) return NULL;
    
    m->rows = rows;
    m->cols = cols;
    m->data = (double*)calloc(rows * cols, sizeof(double));
    
    if (!m->data) {
        free(m);
        return NULL;
    }
    
    return m;
}

void matrix_free(Matrix* m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

void matrix_set(Matrix* m, int row, int col, double value) {
    if (m && row >= 0 && row < m->rows && col >= 0 && col < m->cols) {
        m->data[row * m->cols + col] = value;
    }
}

double matrix_get(Matrix* m, int row, int col) {
    if (m && row >= 0 && row < m->rows && col >= 0 && col < m->cols) {
        return m->data[row * m->cols + col];
    }
    return 0.0;
}

void matrix_randomize(Matrix* m) {
    if (!m) return;
    
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = (double)rand() / RAND_MAX;
    }
}

void matrix_zeros(Matrix* m) {
    if (!m) return;
    
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = 0.0;
    }
}

void matrix_print(Matrix* m) {
    if (!m) return;
    
    printf("Matrix (%d x %d):\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%8.4f ", matrix_get(m, i, j));
        }
        printf("\n");
    }
}

// Naive matrix multiplication: C = A * B
// Triple nested loop - most straightforward implementation
int matrix_multiply_naive(Matrix* A, Matrix* B, Matrix* C) {
    // Check dimensions: A (M x N) * B (N x P) = C (M x P)
    if (!A || !B || !C) return -1;
    if (A->cols != B->rows) return -1;
    if (C->rows != A->rows || C->cols != B->cols) return -1;
    
    int M = A->rows;
    int N = A->cols;
    int P = B->cols;
    
    // Naive triple loop: O(M*N*P)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                // C[i][j] += A[i][k] * B[k][j]
                sum += matrix_get(A, i, k) * matrix_get(B, k, j);
            }
            matrix_set(C, i, j, sum);
        }
    }
    
    return 0;
}
