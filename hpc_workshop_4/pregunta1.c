#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int n_vec = 10000000;   // tamaño del vector grande
    int n_mat = 1000;       // tamaño de la matriz
    int num_threads[] = {1, 2, 4, 8, 16, 32}; // hilos a probar

    // Reservar memoria
    double *A = malloc(n_vec * sizeof(double));
    double *B = malloc(n_vec * sizeof(double));
    double *C = malloc(n_vec * sizeof(double));
    double *x = malloc(n_mat * sizeof(double));
    double *y = malloc(n_mat * sizeof(double));
    double **M = malloc(n_mat * sizeof(double*));

    // Inicializar vectores
    for (int i = 0; i < n_vec; i++) {
        A[i] = i * 1.0;
        B[i] = i * 2.0;
    }

    // Inicializar matriz y vector
    for (int i = 0; i < n_mat; i++) {
        x[i] = 1.0;
        y[i] = 0.0;
        M[i] = malloc(n_mat * sizeof(double));
        for (int j = 0; j < n_mat; j++)
            M[i][j] = (i == j) ? 1.0 : 0.0; // matriz identidad
    }

    double start, end;

    // --- Vector addition sequential ---
    start = omp_get_wtime();
    for (int i = 0; i < n_vec; i++) C[i] = A[i] + B[i];
    end = omp_get_wtime();
    printf("Vector addition sequential: %f seconds\n", end - start);

    // --- Vector addition parallel (different threads and chunks) ---
    for (int t = 0; t < 6; t++) {
        omp_set_num_threads(num_threads[t]);

        // Static schedule
        start = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_vec; i++) C[i] = A[i] + B[i];
        end = omp_get_wtime();
        printf("Vector addition parallel (%d threads, static): %f seconds\n", num_threads[t], end - start);

        // Dynamic schedule, chunk 100
        start = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic, 100)
        for (int i = 0; i < n_vec; i++) C[i] = A[i] + B[i];
        end = omp_get_wtime();
        printf("Vector addition parallel (%d threads, dynamic chunk 100): %f seconds\n", num_threads[t], end - start);

        // Dynamic schedule, chunk 10
        start = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic, 10)
        for (int i = 0; i < n_vec; i++) C[i] = A[i] + B[i];
        end = omp_get_wtime();
        printf("Vector addition parallel (%d threads, dynamic chunk 10): %f seconds\n", num_threads[t], end - start);
    }

    // --- Matrix-vector multiplication sequential ---
    for (int i = 0; i < n_mat; i++) y[i] = 0.0; // reiniciar y
    start = omp_get_wtime();
    for (int i = 0; i < n_mat; i++)
        for (int j = 0; j < n_mat; j++)
            y[i] += M[i][j] * x[j];
    end = omp_get_wtime();
    printf("Matrix-vector sequential: %f seconds\n", end - start);

    // --- Matrix-vector multiplication parallel ---
    for (int t = 0; t < 6; t++) {
        omp_set_num_threads(num_threads[t]);

        // Static schedule
        for (int i = 0; i < n_mat; i++) y[i] = 0.0;
        start = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_mat; i++)
            for (int j = 0; j < n_mat; j++)
                y[i] += M[i][j] * x[j];
        end = omp_get_wtime();
        printf("Matrix-vector parallel (%d threads, static): %f seconds\n", num_threads[t], end - start);

        // Dynamic schedule, chunk 10
        for (int i = 0; i < n_mat; i++) y[i] = 0.0;
        start = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic, 10)
        for (int i = 0; i < n_mat; i++)
            for (int j = 0; j < n_mat; j++)
                y[i] += M[i][j] * x[j];
        end = omp_get_wtime();
        printf("Matrix-vector parallel (%d threads, dynamic chunk 10): %f seconds\n", num_threads[t], end - start);

        // Dynamic schedule, chunk 100
        for (int i = 0; i < n_mat; i++) y[i] = 0.0;
        start = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic, 100)
        for (int i = 0; i < n_mat; i++)
            for (int j = 0; j < n_mat; j++)
                y[i] += M[i][j] * x[j];
        end = omp_get_wtime();
        printf("Matrix-vector parallel (%d threads, dynamic chunk 100): %f seconds\n", num_threads[t], end - start);
    }

    // Free memory
    free(A); free(B); free(C); free(x); free(y);
    for (int i = 0; i < n_mat; i++) free(M[i]);
    free(M);

    return 0;
}