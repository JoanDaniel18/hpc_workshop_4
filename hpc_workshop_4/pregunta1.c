#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int n_vec = 10000000;   // tamaño del vector grande
    int n_mat = 1000;      // tamaño de la matriz 
    double *A = malloc(n_vec * sizeof(double));
    double *B = malloc(n_vec * sizeof(double));
    double *C = malloc(n_vec * sizeof(double));
    double *x = malloc(n_mat * sizeof(double));
    double *y = malloc(n_mat * sizeof(double));
    double **M = malloc(n_mat * sizeof(double*));

    for (int i = 0; i < n_vec; i++) {
        A[i] = i * 1.0;
        B[i] = i * 2.0;
    }

    for (int i = 0; i < n_mat; i++) {
        x[i] = 1.0;
        y[i] = 0.0;
        M[i] = malloc(n_mat * sizeof(double));
        for (int j = 0; j < n_mat; j++) M[i][j] = (i == j) ? 1.0 : 0.0; // identidad
    }

    // Vector addition secuencial
    double start = omp_get_wtime();
    for (int i = 0; i < n_vec; i++) C[i] = A[i] + B[i];
    double end = omp_get_wtime();
    printf("Vector addition secuencial: %f segundos\n", end - start);
    #pragma omp parallel 
    printf("%d\n", omp_get_num_threads());

    // Vector addition paralelo
    start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < n_vec; i++) C[i] = A[i] + B[i];
    end = omp_get_wtime();
    printf("Vector addition paralelo: %f segundos\n", end - start);

    // Matrix-vector secuencial
    start = omp_get_wtime();
    for (int i = 0; i < n_mat; i++)
        for (int j = 0; j < n_mat; j++)
            y[i] += M[i][j] * x[j];
    end = omp_get_wtime();
    printf("Matrix-Vector secuencial: %f segundos\n", end - start);

    // Matrix-vector paralelo
    start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < n_mat; i++)
        for (int j = 0; j < n_mat; j++)
            y[i] += M[i][j] * x[j];
    end = omp_get_wtime();
    printf("Matrix-Vector paralelo: %f segundos\n", end - start);

    return 0;
}
