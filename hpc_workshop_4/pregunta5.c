#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <N> <threads>\n", argv[0]);
        return 1;
    }

    long N = atol(argv[1]);
    int threads = atoi(argv[2]);
    if (N <= 0 || threads <= 0) {
        fprintf(stderr, "N y threads deben ser mayores que 0\n");
        return 1;
    }
    omp_set_num_threads(threads);

    double t0, t1;
    long counter;

    // --- CRITICAL ---
    counter = 0;
    t0 = omp_get_wtime();
    #pragma omp parallel for
    for (long i = 0; i < N; i++) {
        #pragma omp critical
        {
            counter++;
        }
    }
    t1 = omp_get_wtime();
    printf("critical,threads=%d,N=%ld,counter=%ld,time=%.6f\n",
           threads, N, counter, t1 - t0);

    // --- ATOMIC ---
    counter = 0;
    t0 = omp_get_wtime();
    #pragma omp parallel for
    for (long i = 0; i < N; i++) {
        #pragma omp atomic
        counter++;
    }
    t1 = omp_get_wtime();
    printf("atomic,threads=%d,N=%ld,counter=%ld,time=%.6f\n",
           threads, N, counter, t1 - t0);

    // --- REDUCTION ---
    long sum = 0;
    t0 = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++) {
        sum += 1;
    }
    t1 = omp_get_wtime();
    printf("reduction,threads=%d,N=%ld,sum=%ld,time=%.6f\n",
           threads, N, sum, t1 - t0);

    return 0;
}