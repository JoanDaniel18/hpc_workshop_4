// mm_openmp.c
// Compilar con:
//     gcc -O3 -fopenmp mm_openmp.c -o mm_openmp

// Probar con distintos t:

//    for t in 1 2 4 8 16 32; do
//      ./mm_openmp 500  $t seq
//      ./mm_openmp 500  $t outer
//      ./mm_openmp 500  $t nested
//    done


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h> // <-- Agrega esta línea

double *alloc_matrix(int n){
    return (double*) malloc(sizeof(double)*n*n);
}

void generate_matrix(int n, double *M){
    for(int i=0;i<n*n;i++) M[i] = (double)(rand() % 2);
}

void print_matrix(int n, double *M){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++) printf("%.0f ", M[i*n + j]);
        printf("\n");
    }
}

// Secuencial
void mm_seq(int n, double *A, double *B, double *C){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            double s=0.0;
            for(int k=0;k<n;k++) s += A[i*n+k]*B[k*n+j];
            C[i*n+j] = s;
        }
    }
}

// OpenMP: paralelizar sobre i
void mm_omp_outer(int n, double *A, double *B, double *C){
    #pragma omp parallel for
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            double s=0.0;
            for(int k=0;k<n;k++) s += A[i*n+k]*B[k*n+j];
            C[i*n+j] = s;
        }
    }
}

// OpenMP: paralelizar sobre i y j (nested)
void mm_omp_nested(int n, double *A, double *B, double *C){
    #pragma omp parallel for collapse(2)
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            double s=0.0;
            for(int k=0;k<n;k++) s += A[i*n+k]*B[k*n+j];
            C[i*n+j] = s;
        }
    }
}

int main(int argc, char **argv){
    if(argc < 4){
        fprintf(stderr,"Usage: %s n threads variant\nvariant: seq | outer | nested\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int threads = atoi(argv[2]);
    char *variant = argv[3];
    srand(0);

    omp_set_num_threads(threads);

    double *A = alloc_matrix(n), *B = alloc_matrix(n), *C = alloc_matrix(n);
    generate_matrix(n, A);
    generate_matrix(n, B);

    double t0 = omp_get_wtime();

    if(strcmp(variant,"seq")==0){
        mm_seq(n,A,B,C);
    }else if(strcmp(variant,"outer")==0){
        mm_omp_outer(n,A,B,C);
    }else if(strcmp(variant,"nested")==0){
        mm_omp_nested(n,A,B,C);
    }else{
        fprintf(stderr,"Unknown variant %s\n", variant);
        return 1;
    }

    double t1 = omp_get_wtime();
    double elapsed = t1 - t0;

    double sum=0.0;
    for(int i=0;i<n*n;i++) sum += C[i];
    printf("variant=%s threads=%d n=%d time=%.6f checksum=%.3f\n",
           variant, threads, n, elapsed, sum);

    free(A); free(B); free(C);
    return 0;
}
