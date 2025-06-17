#include <cstdio>
#include <cuda_runtime.h>

constexpr int N = 1024;               // square matrix size (change as needed)
using dtype = float;                  // easy to compare on host

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

int main() {
    size_t bytes = N * N * sizeof(dtype);
    dtype *hA = (dtype*)malloc(bytes),
          *hB = (dtype*)malloc(bytes),
          *hC = (dtype*)malloc(bytes);

    // host init
    for (int i = 0; i < N * N; ++i) {
        hA[i] = static_cast<dtype>((i % 3) + 1);
        hB[i] = static_cast<dtype>((i % 5) + 1);
    }

    // device alloc & copy
    dtype *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    // launch
    dim3 blk(16, 16);
    dim3 grd((N + blk.x - 1) / blk.x, (N + blk.y - 1) / blk.y);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    matmul_naive<<<grd, blk>>>(dA, dB, dC, N);
    cudaEventRecord(t1);

    cudaEventSynchronize(t1);
    float msec = 0.0f;
    cudaEventElapsedTime(&msec, t0, t1);

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    // quick correctness spot-check (C[0] == sum_{k} A[0,k]*B[k,0])
    dtype gold = 0;
    for (int k = 0; k < N; ++k)
        gold += hA[k] * hB[k * N];
    printf("C[0,0] = %f  (gold %f) — %s\n",
           hC[0], gold, (fabs(hC[0] - gold) < 1e-3) ? "OK" : "MISMATCH");

    double flops = 2.0 * N * N * N;
    double gflops = flops / (msec * 1e6);
    printf("Naive GEMM %dx%d took %.3f ms → %.2f GFLOP/s\n", N, N, msec, gflops);

    // cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}