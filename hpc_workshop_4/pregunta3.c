#include <stdio.h>
#include <math.h>
#include <omp.h>

double heavy_computation(int i) {
    double result = 0.0;
    int iterations = 1000 + (i % 1000);
    for (int j = 0; j < iterations; j++) {
        result += sin(i * 0.001) * cos(j * 0.001);
    }
    return result;
}

int main() {
    int N = 15000;
    double sum = 0.0;
    double start, end;
    double seq_time, par_time;

    printf("=== Activity 3: Uneven workload with OpenMP schedules ===\n\n");

    // --- Sequential version (baseline) ---
    sum = 0.0;
    start = omp_get_wtime();
    for (int i = 0; i < N; i++)
        sum += heavy_computation(i);
    end = omp_get_wtime();
    seq_time = end - start;
    printf("Sequential: %.4f s\n", seq_time);

    // --- Parallel Static ---
    #pragma omp parallel 
    printf("Threads: %d\n", omp_get_num_threads());
    sum = 0.0;
    start = omp_get_wtime();
    #pragma omp parallel for schedule(static) reduction(+:sum)    
    for (int i = 0; i < N; i++)
        sum += heavy_computation(i);
    end = omp_get_wtime();
    par_time = end - start;
    printf("Static schedule: %.4f s, Speedup: %.2f\n", par_time, seq_time / par_time);

    // --- Parallel Dynamic, chunk 4 ---
    sum = 0.0;
    start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 4) reduction(+:sum)
    for (int i = 0; i < N; i++)
        sum += heavy_computation(i);
    end = omp_get_wtime();
    par_time = end - start;
    printf("Dynamic schedule (chunk=4): %.4f s, Speedup: %.2f\n", par_time, seq_time / par_time);

    // --- Parallel Dynamic, chunk 16 ---
    sum = 0.0;
    start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 16) reduction(+:sum)
    for (int i = 0; i < N; i++)
        sum += heavy_computation(i);
    end = omp_get_wtime();
    par_time = end - start;
    printf("Dynamic schedule (chunk=16): %.4f s, Speedup: %.2f\n", par_time, seq_time / par_time);

    // --- Parallel Guided ---
    sum = 0.0;
    start = omp_get_wtime();
    #pragma omp parallel for schedule(guided) reduction(+:sum)
    for (int i = 0; i < N; i++)
        sum += heavy_computation(i);
    end = omp_get_wtime();
    par_time = end - start;
    printf("Guided schedule: %.4f s, Speedup: %.2f\n", par_time, seq_time / par_time);
    return 0;
}