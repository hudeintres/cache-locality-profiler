#ifndef MATRIX_H
#define MATRIX_H

// Matrix structure
typedef struct {
    double* data;
    int rows;
    int cols;
} Matrix;

// Create a matrix with given dimensions
Matrix* matrix_create(int rows, int cols);

// Free matrix memory
void matrix_free(Matrix* m);

// Set matrix element
void matrix_set(Matrix* m, int row, int col, double value);

// Get matrix element
double matrix_get(Matrix* m, int row, int col);

// Initialize matrix with random values
void matrix_randomize(Matrix* m);

// Initialize matrix with zeros
void matrix_zeros(Matrix* m);

// Print matrix (for debugging)
void matrix_print(Matrix* m);

// Naive matrix multiplication: C = A * B
// Returns 0 on success, -1 on dimension mismatch
int matrix_multiply_naive(Matrix* A, Matrix* B, Matrix* C);

#endif // MATRIX_H
