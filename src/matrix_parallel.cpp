#include "matrix.h"

#include <algorithm>
#include <thread>
#include <vector>

namespace {
int normalize_thread_count(int num_threads, int max_rows) {
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
    }
    if (num_threads <= 0) {
        num_threads = 1;
    }
    return std::min(num_threads, std::max(1, max_rows));
}
}

int matrix_multiply_naive_parallel(Matrix* A, Matrix* B, Matrix* C, int num_threads) {
    if (!A || !B || !C) return -1;
    if (A->cols != B->rows) return -1;
    if (C->rows != A->rows || C->cols != B->cols) return -1;

    int M = A->rows;
    int N = A->cols;
    int P = B->cols;

    int threads = normalize_thread_count(num_threads, M);
    matrix_zeros(C);

    auto worker = [A, B, C, N, P](int row_start, int row_end) {
        for (int i = row_start; i < row_end; ++i) {
            for (int j = 0; j < P; ++j) {
                double sum = 0.0;
                int a_base = i * A->cols;
                for (int k = 0; k < N; ++k) {
                    sum += A->data[a_base + k] * B->data[k * B->cols + j];
                }
                C->data[i * C->cols + j] = sum;
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));

    int rows_per_thread = (M + threads - 1) / threads;
    int row_start = 0;
    for (int t = 0; t < threads && row_start < M; ++t) {
        int row_end = std::min(M, row_start + rows_per_thread);
        workers.emplace_back(worker, row_start, row_end);
        row_start = row_end;
    }

    for (auto& worker_thread : workers) {
        worker_thread.join();
    }

    return 0;
}

int matrix_multiply_transpose_parallel(Matrix* A, Matrix* B, Matrix* C, int num_threads) {
    if (!A || !B || !C) return -1;
    if (A->cols != B->rows) return -1;
    if (C->rows != A->rows || C->cols != B->cols) return -1;

    int M = A->rows;
    int N = A->cols;
    int P = B->cols;

    Matrix* B_T = matrix_create(P, N);
    if (!B_T) return -1;

    for (int i = 0; i < B->rows; ++i) {
        for (int j = 0; j < B->cols; ++j) {
            B_T->data[j * B_T->cols + i] = B->data[i * B->cols + j];
        }
    }

    int threads = normalize_thread_count(num_threads, M);
    matrix_zeros(C);

    auto worker = [A, B_T, C, N, P](int row_start, int row_end) {
        for (int i = row_start; i < row_end; ++i) {
            int a_base = i * A->cols;
            for (int j = 0; j < P; ++j) {
                double sum = 0.0;
                int b_base = j * B_T->cols;
                for (int k = 0; k < N; ++k) {
                    sum += A->data[a_base + k] * B_T->data[b_base + k];
                }
                C->data[i * C->cols + j] = sum;
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));

    int rows_per_thread = (M + threads - 1) / threads;
    int row_start = 0;
    for (int t = 0; t < threads && row_start < M; ++t) {
        int row_end = std::min(M, row_start + rows_per_thread);
        workers.emplace_back(worker, row_start, row_end);
        row_start = row_end;
    }

    for (auto& worker_thread : workers) {
        worker_thread.join();
    }

    matrix_free(B_T);
    return 0;
}

int matrix_multiply_blocked_parallel(Matrix* A, Matrix* B, Matrix* C, int block_size, int num_threads) {
    if (!A || !B || !C) return -1;
    if (A->cols != B->rows) return -1;
    if (C->rows != A->rows || C->cols != B->cols) return -1;

    int M = A->rows;
    int N = A->cols;
    int P = B->cols;

    int BLOCK;
    if (block_size <= 0) {
        int l1_size = get_l1_cache_size();
        int max_elements = l1_size / (4 * sizeof(double));
        BLOCK = 1;
        while (BLOCK * BLOCK <= max_elements && BLOCK < 128) {
            BLOCK *= 2;
        }
        BLOCK /= 2;
        if (BLOCK < 16) BLOCK = 16;
    } else {
        BLOCK = block_size;
    }

    matrix_zeros(C);

    int threads = normalize_thread_count(num_threads, M);
    int rows_per_thread = (M + threads - 1) / threads;

    auto worker = [A, B, C, N, P, BLOCK](int row_start, int row_end) {
        double* a_data = A->data;
        double* b_data = B->data;
        double* c_data = C->data;
        int a_stride = A->cols;
        int b_stride = B->cols;
        int c_stride = C->cols;

        for (int ii = row_start; ii < row_end; ii += BLOCK) {
            int i_max = std::min(row_end, ii + BLOCK);

            for (int kk = 0; kk < N; kk += BLOCK) {
                int k_max = std::min(N, kk + BLOCK);

                for (int jj = 0; jj < P; jj += BLOCK) {
                    int j_max = std::min(P, jj + BLOCK);

                    for (int i = ii; i < i_max; ++i) {
                        int a_base = i * a_stride;
                        int c_base = i * c_stride;
                        for (int k = kk; k < k_max; ++k) {
                            double a_val = a_data[a_base + k];
                            int b_base = k * b_stride;
                            for (int j = jj; j < j_max; ++j) {
                                c_data[c_base + j] += a_val * b_data[b_base + j];
                            }
                        }
                    }
                }
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));

    int row_start = 0;
    for (int t = 0; t < threads && row_start < M; ++t) {
        int row_end = std::min(M, row_start + rows_per_thread);
        workers.emplace_back(worker, row_start, row_end);
        row_start = row_end;
    }

    for (auto& worker_thread : workers) {
        worker_thread.join();
    }

    return 0;
}
