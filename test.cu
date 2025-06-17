#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cassert>

constexpr int Mb = 128;
constexpr int Nb = 64;

__global__ void init_half_matrix(__half* A) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= Mb) return;
    for (int col = 0; col < Nb; col++) {
        A[row * Nb + col] = __float2half(static_cast<float>(row * Nb + col));
    }
}

__global__ void check_tmem_load(uint32_t* tmem32, float* C, int Nb) {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    for (int i = 0; i < Nb; i++) {
        uint32_t r0;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
            : "=r"(r0)
            : "l"(reinterpret_cast<uint64_t>(tmem32) + ((warp * 32ULL) << 16) + i));
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        C[threadIdx.x * Nb + i] = __uint_as_float(r0);
    }
}

int main() {
    __half* A_dev;
    float* C_dev;
    float* C_host = new float[Mb * Nb];

    cudaMalloc(&A_dev, Mb * Nb * sizeof(__half));
    cudaMalloc(&C_dev, Mb * Nb * sizeof(float));

    // Launch to fill A_dev with known values
    init_half_matrix<<<(Mb + 31)/32, 32>>>(A_dev);
    cudaDeviceSynchronize();

    // Use the same address for fake tmem handle (simulation only)
    uint32_t* tmem32 = reinterpret_cast<uint32_t*>(A_dev);
    check_tmem_load<<<1, 128>>>(tmem32, C_dev, Nb);
    cudaMemcpy(C_host, C_dev, Mb * Nb * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate some entries
    for (int i = 0; i < 16; ++i) {
        printf("C[%d] = %.1f\n", i, C_host[i]);
    }

    cudaFree(A_dev);
    cudaFree(C_dev);
    delete[] C_host;
    return 0;
}