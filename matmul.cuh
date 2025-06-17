#include <cstdio>

__global__ void hello_asm(unsigned *out)
{
    unsigned tid;               // scratch register
    // One PTX instruction just to prove inline-PTX works
    asm volatile ("mov.u32 %0, %%tid.x;" : "=r"(tid));
    if (threadIdx.x == 0 && blockIdx.x == 0)   // single write
        out[0] = tid;          // should be 0
}

int main() {
    unsigned *d, h = 1234;
    cudaMalloc(&d, sizeof(unsigned));
    hello_asm<<<1, 32>>>(d);
    cudaMemcpy(&h, d, sizeof(unsigned), cudaMemcpyDeviceToHost);
    printf("Inline-PTX says tid.x = %u (expect 0)\n", h);
    return 0;
}