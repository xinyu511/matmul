#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

using dtype = float;
constexpr int REPS = 10;

__global__ void matmul_naive(const dtype* A,
                             const dtype* B,
                             dtype* C,
                             int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;

    dtype acc = 0;
    for (int k = 0; k < n; ++k)
        acc += A[row * n + k] * B[k * n + col];

    C[row * n + col] = acc;
}

int run_naive_benchmark(int N)
{
    printf("--------------------  N=%d  --------------------\n", N);
    size_t bytes = size_t(N) * N * sizeof(dtype);

    dtype *hA = (dtype*)malloc(bytes),
          *hB = (dtype*)malloc(bytes),
          *hC = (dtype*)malloc(bytes),
          *hC_ref = (dtype*)malloc(bytes);

    // Initialize host matrices
    for (int i = 0; i < N * N; ++i) {
        hA[i] = static_cast<dtype>((i % 3) + 1);
        hB[i] = static_cast<dtype>((i % 5) + 1);
    }

    // Reference (CPU) correctness check for small N
    bool do_check = (N <= 2048);
    if (do_check) {
        printf("Running CPU reference ...\n");
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                dtype sum = 0;
                for (int k = 0; k < N; ++k)
                    sum += hA[i * N + k] * hB[k * N + j];
                hC_ref[i * N + j] = sum;
            }
    }

    // Device memory allocation
    dtype *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    // Kernel config
    dim3 blk(16, 16);
    dim3 grd((N + 15) / 16, (N + 15) / 16);

    // Warm-up
    matmul_naive<<<grd, blk>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    // Timed run
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < REPS; ++r)
        matmul_naive<<<grd, blk>>>(dA, dB, dC, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float msec = 0.0f;
    cudaEventElapsedTime(&msec, t0, t1);
    msec /= REPS;

    // Copy back and validate
    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    if (do_check) {
        double max_err = 0.0, avg_err = 0.0;
        for (size_t i = 0; i < size_t(N) * N; ++i) {
            double err = fabs(double(hC[i]) - double(hC_ref[i]));
            max_err = std::max(max_err, err);
            avg_err += err;
        }
        avg_err /= (N * N);
        printf("Max err %.5g  Avg err %.5g\n", max_err, avg_err);
    } else {
        printf("(skipped reference check; matrix too big)\n");
    }

    // Performance report
    double flops = 2.0 * N * N * N;
    double gflops = flops / (msec * 1e6);
    printf("Naive GEMM time %.3f ms  →  %.2f GFLOP/s\n", msec, gflops);

    // Cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hC_ref);
    return 0;
}

int main()
{
    for (int N : {1024, 2048, 4096, 8192}) {
        run_naive_benchmark(N);
    }
    return 0;
}