#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int N = 10000000;  // tamaño del vector (10 millones de elementos)
    double *A = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) A[i] = 1.0;  // inicializamos todo a 1.0

    double sum = 0.0;
    double start, end;

    // ---------------------------
    // Versión correcta con reduction
    // ---------------------------
    start = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += A[i];
    }
    end = omp_get_wtime();
    printf("Suma CORRECTA con reduction = %.2f (tiempo: %f s)\n", sum, end - start);

    // ---------------------------
    // Versión incorrecta con shared (condición de carrera)
    // ---------------------------
    sum = 0.0;
    start = omp_get_wtime();
    #pragma omp parallel for shared(sum)
    for (int i = 0; i < N; i++) {
        sum += A[i];  // aquí varios hilos escriben a la vez en sum → resultado incorrecto
    }
    end = omp_get_wtime();
    printf("Suma INCORRECTA con shared = %.2f (tiempo: %f s)\n", sum, end - start);

    // ---------------------------
    // Versión con private (también incorrecta si no acumulamos bien)
    // ---------------------------
    sum = 0.0;
    start = omp_get_wtime();
    #pragma omp parallel for private(sum)
    for (int i = 0; i < N; i++) {
        sum += A[i];  // cada hilo tiene su propia copia de sum, pero no se combina al final
    }
    end = omp_get_wtime();
    printf("Suma INCORRECTA con private = %.2f (tiempo: %f s)\n", sum, end - start);

    free(A);
    return 0;
}
