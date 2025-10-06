#include <stdio.h>
#include <omp.h>
#include <unistd.h> // para sleep (simula trabajo)

void task1() { sleep(2); }
void task2() { sleep(3); }
void task3() { sleep(1); }
void task4() { sleep(4); }

int main() {
    double start, end;

    // --- 2 sections ---
    start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        task1();
        #pragma omp section
        task2();
    }
    end = omp_get_wtime();
    printf("2 sections: %f seconds\n", end - start);

    // --- 3 sections ---
    start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        task1();
        #pragma omp section
        task2();
        #pragma omp section
        task3();
    }
    end = omp_get_wtime();
    printf("3 sections: %f seconds\n", end - start);

    // --- 4 sections ---
    start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        task1();
        #pragma omp section
        task2();
        #pragma omp section
        task3();
        #pragma omp section
        task4();
    }
    end = omp_get_wtime();
    printf("4 sections: %f seconds\n", end - start);

    return 0;
}