#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CPU matrix multiplication
void matrix_multiply_cpu(int* A, int* B, int* C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

// GPU matrix multiplication kernel
__global__ void matrix_multiply_gpu(int* A, int* B, int* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

int main() {
    // Matrix dimensions m=rows of matrix A, n=columns of matrix A & the number of rows of matrix B, p=columns of matrix B
    int m = 1 << 8, n = 1 << 8, p = 1 << 8;

    // Host matrices
    int* A, * B, * C_cpu;
    A = (int*)malloc(m * n * sizeof(int));
    B = (int*)malloc(n * p * sizeof(int));
    C_cpu = (int*)malloc(m * p * sizeof(int));

    // Initialize matrices
    srand(time(NULL)); // Seed for random number generation
    for (int i = 0; i < m * n; i++) {
        A[i] = rand() % 10;
    }
    for (int i = 0; i < n * p; i++) {
        B[i] = rand() % 10;
    }

    // CPU matrix multiplication
    clock_t start_cpu, end_cpu;
    double cpu_time_used;
    start_cpu = clock();
    matrix_multiply_cpu(A, B, C_cpu, m, n, p);
    end_cpu = clock();
    cpu_time_used = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    printf("CPU Execution Time: %f seconds\n", cpu_time_used);

    // GPU matrix multiplication
    int* A_gpu, * B_gpu, * C_gpu;
    cudaMalloc((void**)&A_gpu, m * n * sizeof(int));
    cudaMalloc((void**)&B_gpu, n * p * sizeof(int));
    cudaMalloc((void**)&C_gpu, m * p * sizeof(int));
    cudaMemcpy(A_gpu, A, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, n * p * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid((p + 15) / 16, (m + 15) / 16, 1);
    dim3 dimBlock(64, 64, 1);

    cudaEvent_t start_gpu, end_gpu;
    float gpu_time_ms;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);
    cudaEventRecord(start_gpu);
    matrix_multiply_gpu << <dimGrid, dimBlock >> > (A_gpu, B_gpu, C_gpu, m, n, p);
    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);
    cudaEventElapsedTime(&gpu_time_ms, start_gpu, end_gpu);
    printf("GPU Execution Time: %f milliseconds\n", gpu_time_ms);

    // Copy result from GPU to host
    int* C_gpu_host = (int*)malloc(m * p * sizeof(int));
    cudaMemcpy(C_gpu_host, C_gpu, m * p * sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory
    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu_host);
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);

    return 0;
}
