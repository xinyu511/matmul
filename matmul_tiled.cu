/**********************************************************************
 * 16×16 shared-memory GEMM benchmark for Blackwell (sm_100 / sm_100a)
 * -------------------------------------------------------------------
 *  – dtype = float  (easy to compare; change to bf16 if you like)
 *  – TILE = 16      (matches the kernel we developed)
 *  – runs N = 1 024, 2 048, 4 096, 8 192, 16 384
 *********************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <omp.h>

using dtype        = float;
constexpr int TILE = 16;
constexpr int REPS = 10;           // timed iterations / size

// ------------------------------------------------------------------
// Device kernel: 16×16 tiled GEMM (identical to the earlier example)
// ------------------------------------------------------------------
__global__ void matmul_tiled(const dtype* __restrict__ A,
                             const dtype* __restrict__ B,
                             dtype* __restrict__ C,
                             int n)
{
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    extern __shared__ dtype smem[];
    dtype* As = smem;                     // 256 elements
    dtype* Bs = smem + TILE * TILE;       // 256 elements

    dtype acc = 0;

    for (int t = 0; t < n; t += TILE)
    {
        As[threadIdx.y * TILE + threadIdx.x] = A[row * n + (t + threadIdx.x)];
        Bs[threadIdx.y * TILE + threadIdx.x] = B[(t + threadIdx.y) * n + col];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y * TILE + k] * Bs[k * TILE + threadIdx.x];

        __syncthreads();
    }
    C[row * n + col] = acc;
}

// ------------------------------------------------------------------
// Naïve CPU GEMM for reference (OpenMP-parallel)
// ------------------------------------------------------------------
void cpu_gemm(const dtype* A, const dtype* B, dtype* C,
              int M, int N, int K)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
        {
            dtype sum = 0;
            for (int k = 0; k < K; ++k)
                sum += A[i*K + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

// ------------------------------------------------------------------
// Benchmark one size
// ------------------------------------------------------------------
int run_benchmark(int M, int N, int K)
{
    std::cout << "--------------------  M=" << M
              << " N=" << N
              << " K=" << K << "  --------------------\n";

    /* ---- host alloc & init ---- */
    const size_t bytesA = size_t(M) * K * sizeof(dtype);
    const size_t bytesB = size_t(K) * N * sizeof(dtype);
    const size_t bytesC = size_t(M) * N * sizeof(dtype);

    dtype *hA = static_cast<dtype*>(malloc(bytesA));
    dtype *hB = static_cast<dtype*>(malloc(bytesB));
    dtype *hC = static_cast<dtype*>(malloc(bytesC));
    dtype *hC_ref = static_cast<dtype*>(malloc(bytesC));

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    for (size_t i = 0; i < size_t(M)*K; ++i) hA[i] = dis(gen);
    for (size_t i = 0; i < size_t(K)*N; ++i) hB[i] = dis(gen);

    /* ---- optional CPU reference (skip if size too big) ---- */
    const bool do_check = (M <= 8192);
    if (do_check)
    {
        std::cout << "Running CPU reference ...\n";
        cpu_gemm(hA, hB, hC_ref, M, N, K);
    }

    /* ---- device alloc & upload ---- */
    dtype *dA, *dB, *dC;
    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC);
    cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice);

    dim3 blk(TILE, TILE);
    dim3 grd((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    size_t shmem = 2 * TILE * TILE * sizeof(dtype);

    /* ---- warm-up ---- */
    matmul_tiled<<<grd, blk, shmem>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    /* ---- timed loop ---- */
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < REPS; ++r)
        matmul_tiled<<<grd, blk, shmem>>>(dA, dB, dC, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float msec;
    cudaEventElapsedTime(&msec, t0, t1);
    msec /= REPS;                                 // per-iteration

    /* ---- copy back & (optionally) check ---- */
    cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost);

    if (do_check)
    {
        double max_err = 0, avg_err = 0;
        for (size_t i = 0; i < size_t(M)*N; ++i)
        {
            double err = std::abs(double(hC[i]) - double(hC_ref[i]));
            max_err = std::max(max_err, err);
            avg_err += err;
        }
        avg_err /= (double(M)*N);
        std::cout << "Max err " << max_err
                  << "  Avg err " << avg_err << "\n";
    }
    else
        std::cout << "(skipped reference check; matrix too big)\n";

    /* ---- performance ---- */
    double flops  = 2.0 * double(M) * N * K;
    double tflops = (flops / (msec * 1e6)) / 1e3;
    std::cout << "Avg kernel time " << msec << " ms  →  "
              << tflops << " TFLOP/s\n";

    /* ---- cleanup ---- */
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hC_ref);
    return 0;
}

// ------------------------------------------------------------------
// Main driver
// ------------------------------------------------------------------
int main()
{
    for (int N : {8192, 16384})
        run_benchmark(N, N, N);
    return 0;
}

/* -----------------------------------------------------------------
   Build:
       nvcc matmul_tiled_benchmark.cu -std=c++20 -O3 -arch=sm_100 -o matmul_tiled_bench
   (or use -arch=sm_100a if that’s what you’ve been using with ThunderKittens)

   Run:
       ./matmul_tiled_bench
------------------------------------------------------------------*/