// nvcc matmul_tmem.cu -std=c++20 -O3 -arch=sm_100a -Xcompiler -fopenmp -o matmul_tmem [-DDEBUG_UMMA]
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>
#include <random>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <fstream>


struct ProblemSize {
    int M, N, K;
};
__device__ __forceinline__
int swizzle(int r, int c) {
    return ((((r * 8) + (c >> 3)) ^ ((((r * 8) + (c >> 3)) & 56) >> 3)) << 3) | (c & 7);
}
constexpr int Mb = 128, Nb = 128, Kb = 64;
using  Tab  = __half;
using  Tacc = float;

#ifdef DEBUG_UMMA
#   define KDBG(fmt, ...)                                                        \
        do {                                                                     \
            if (blockIdx.x == 0 && threadIdx.x == 0)                              \
                printf("[blk0] " fmt "\n", ##__VA_ARGS__);                       \
        } while (0)
#else
#   define KDBG(fmt, ...)  ((void)0)
#endif

union InstrDescriptor
{
  uint32_t desc_;

  struct {
    // Bitfield implementation avoids the need for shifts in assignment
    uint16_t sparse_id2_    : 2,  // bit [ 0, 2) : Sparse meta data id2
             sparse_flag_   : 1,  // bit [ 2, 3) : 0 = dense. 1 = sparse. 1 value valid only for F32F16/S8/MXF8F6F4
             saturate_      : 1,  // bit [ 3, 4) : 0 = no saturate. 1 = saturate. 1 value valid only for S8
             c_format_      : 2,  // bit [ 4, 6) : 0 = F16. 1 = F32, 2 = S32
                            : 1,  //
             a_format_      : 3,  // bit [ 7,10) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1 signed 8 bit. Boolean MMA: 0 Boolean
             b_format_      : 3,  // bit [10,13) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1 signed 8 bit. Boolean MMA: 0 Boolean
             a_negate_      : 1,  // bit [13,14) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format and MXF8F6F4Format
             b_negate_      : 1,  // bit [14,15) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format and MXF8F6F4Format
             a_major_       : 1;  // bit [15,16) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
    uint16_t b_major_       : 1,  // bit [16,17) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
             n_dim_         : 6,  // bit [17,23) : 3 LSBs not included. Valid values range from 1 (N=8) to 32 (N=256).  All values are not valid for all instruction formats
                            : 1,  //
             m_dim_         : 5,  // bit [24,29) : 4 LSBs not included. Valid values are: 4 (M=64), 8 (M=128), 16 (M=256)
                            : 1,  //
             max_shift_     : 2;  // bit [30,32) : Maximum shift for WS instruction. Encoded as follows: 0 = no shift, 1 = maximum shift of 8, 2 = maximum shift of 16, 3 = maximum shift of 32.
  };

  // Decay to a uint32_t
  __host__ __device__ constexpr explicit
  operator uint32_t() const noexcept { return desc_; }
};

__forceinline__ __device__ void ptx_tcgen05_encode_instr_descriptor(uint32_t* desc, int M, int N, int d_format,
                                            int a_format, int b_format, bool trans_a, bool trans_b,
                                            bool neg_a, bool neg_b, bool sat_d, bool is_sparse) {
  InstrDescriptor _desc;

  _desc.a_format_ = uint8_t(a_format);
  _desc.b_format_ = uint8_t(b_format);
  _desc.c_format_ = uint8_t(d_format);

  _desc.m_dim_ = (M >> 4);
  _desc.n_dim_ = (N >> 3);

  _desc.a_major_ = static_cast<uint8_t>(trans_a);
  _desc.b_major_ = static_cast<uint8_t>(trans_b);

  _desc.a_negate_ = static_cast<uint8_t>(neg_a);
  _desc.b_negate_ = static_cast<uint8_t>(neg_b);
  _desc.saturate_ = static_cast<uint8_t>(sat_d);

  _desc.sparse_flag_ = is_sparse;
  _desc.sparse_id2_  = 0;                          // should modify in sparse case

  _desc.max_shift_ = uint8_t(0);                   // WS not used

  *desc = (uint32_t)_desc;
}



union SmemDescriptor
{
  uint64_t desc_ = 0;
  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2;                     // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    uint16_t leading_byte_offset_ : 14, : 2;               // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    uint16_t stride_byte_offset_ : 14, version_ : 2;       // 14 bits [0,14), 2 bits [14,16)
    // base_offset, bit [49,52). leading_byte_offset_mode, bit [52,53).
    uint8_t : 1, base_offset_ : 3, lbo_mode_ : 1, : 3;     // 1 bit unused, 3 bits [1,4), 1 bit [4,5), 3 bits unused
    // layout type, bit [61,64), SWIZZLE_NONE matrix descriptor = 0, SWIZZLE_128B matrix descriptor = 2, SWIZZLE_64B descriptor = 4, SWIZZLE_32B descriptor = 6, SWIZZLE_128B_BASE32B = 1, N/A = 3, N/A = 5, N/A = 7
    uint8_t : 5, layout_type_ : 3;                         // 6 bits unused, 3 bits [5,8)
  };
  // Seperate the field, as we may only update one part of desc
  struct {
    uint32_t lo;
    uint32_t hi;
  };

  // Decay to a uint64_t
  __host__ __device__ constexpr
  operator uint64_t() const noexcept { return desc_; }
};

__forceinline__ __device__ void ptx_tcgen05_encode_matrix_descriptor(uint64_t* desc, void* addr, int ldo, int sdo, int swizzle) {
  SmemDescriptor _desc;

  _desc.version_ = 1;
  _desc.lbo_mode_ = 0;

  switch (swizzle) {
    case 0: _desc.layout_type_ = uint8_t(0); break; // No swizzle
    case 1: _desc.layout_type_ = uint8_t(6); break; // 32B swizzle
    case 2: _desc.layout_type_ = uint8_t(4); break; // 64B swizzle
    case 3: _desc.layout_type_ = uint8_t(2); break; // 128B swizzle
    case 4: _desc.layout_type_ = uint8_t(1); break; // 128B_base32B swizzle
  }

  uint32_t start_address = __cvta_generic_to_shared(addr);
  _desc.start_address_ = static_cast<uint16_t>(start_address >> 4);

  constexpr uint8_t base_offset = 0;
  _desc.base_offset_ = base_offset;

  _desc.stride_byte_offset_  = static_cast<uint32_t>(sdo);
  _desc.leading_byte_offset_ = static_cast<uint32_t>(ldo);

  *desc = (uint64_t)_desc;
}
__forceinline__ __device__ uint32_t get_tmem_addr(uint32_t idx, int row_offset, int col_offset) {
  int col_idx = idx & 0xFFFF;
  int row_idx = (idx >> 16) & 0xFFFF;
  col_idx += col_offset;
  row_idx += row_offset;
  col_idx = col_idx & 0xFFFF;
  row_idx = row_idx & 0xFFFF;

  uint32_t new_idx = (row_idx << 16) | col_idx;
  return new_idx;
}

__device__ __forceinline__ uint32_t matrix_encode(uint32_t x) { return x>> 4; }

__device__ inline uint64_t make_st_desc(const Tab* smem) { 
    SmemDescriptor desc; 
    desc.start_address_ = matrix_encode(static_cast<uint32_t>(__cvta_generic_to_shared(smem)));
    desc.leading_byte_offset_ = 1;
    desc.stride_byte_offset_ = 64;
    desc.base_offset_ = 0;
    desc.lbo_mode_ = 0;
    desc.layout_type_ = 2;
    desc.version_ = 1;
    return desc;
}

//copied from thunderkittens
template<int phaseBit>
__device__ inline void mbarrier_wait(void* bar_b64)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar_b64));
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(addr),
        "r"(phaseBit)
    );
}

extern "C" __global__ void __launch_bounds__(128) umma_tile(half* __restrict__ A, half* __restrict__ B, float* __restrict__ C, ProblemSize  prob);
extern "C" __global__ void __launch_bounds__(128) umma_tile(half* __restrict__ A, half* __restrict__ B, float* __restrict__ C, ProblemSize  prob) {
    KDBG("kernel entry");

    int tileM = blockIdx.y;      // 0 … 7
    int tileN = blockIdx.x;      // 0 … 7
    int tileK = blockIdx.z;      // 0 … 15

    int row0 = tileM * Mb;
    int col0 = tileN * Nb;
    int k0   = tileK * Kb;
    __shared__ alignas(256) half A_smem[8192];
    __shared__ alignas(256) half B_smem[8192];
    // extern __shared__ Tab sm[];
    // Tab* A_smem = sm;               // 128×64 = 8 KiB
    // Tab* B_smem = sm + Mb*Kb;     
    alignas(64) float reg[128];
    __shared__ alignas(8) uint tmem_addr[1];
    /* ---- allocate 64-column TMEM -------------------------------- */
    alignas(64) uint64_t descA[1];
    alignas(64) uint64_t descB[1];
    alignas(64) uint descI[1];
    unsigned int smem_addr = __cvta_generic_to_shared(tmem_addr);
    if (((int)threadIdx.x) < 32) {
        __asm__ __volatile__(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(smem_addr), "r"(512)
        : "memory"
        );
    }
    __syncthreads();
    for (int i = 0; i < 128; ++i) {
        reg[i] = 0.000000e+00f;
    }
    const int elemsA = Mb * Kb;          // 128 × 64 = 8192 half
    const int elemsB = Nb * Kb;          // 128 × 64 = 8192 half

    for (int i = threadIdx.x;  i < elemsA;  i += blockDim.x) {
        int r = i / Kb;                  // row inside tile
        int c = i - r*Kb;                // col inside tile
        int s = swizzle(r, c);           // shared-mem index
        int g = (row0 + r) * prob.K + (k0 + c);
        A_smem[s] = A[g];
    }

    for (int i = threadIdx.x;  i < elemsB;  i += blockDim.x) {
        int r = i / Kb;
        int c = i - r*Kb;
        int s = swizzle(r, c);
        int g = (col0 + r) * prob.K + (k0 + c);
        B_smem[s] = B[g];
    }
    __syncthreads(); 

    __syncthreads();
    KDBG("tiles copied to SMEM");
    __shared__ unsigned long long sem;
    if (((int)threadIdx.x) == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "l"(&sem));
        uint64_t sem_addr = static_cast<uint64_t>(__cvta_generic_to_shared(&sem));
        ptx_tcgen05_encode_instr_descriptor(descI, 128, 128, 1, 0, 0, false, false, false, false, false, false);
        for (int k = 0; k < 4; ++k) {
        ptx_tcgen05_encode_matrix_descriptor(descA, (&(A_smem[(((k * 2) ^ (((k * 2) & 56) >> 3)) << 3)])), 1, 64, 3);
        ptx_tcgen05_encode_matrix_descriptor(descB, (&(B_smem[(((k * 2) ^ (((k * 2) & 56) >> 3)) << 3)])), 1, 64, 3);
        if (k == 0) {
            
            {
                /* T.ptx_tcgen05_mma() */
                asm volatile(
                    "{\n"
                    ".reg .pred p;\n"
                    "setp.eq.u32 p, 1, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "
                    "{%5, %6, %7, %8}, p;\n"
                    "}\n"
                    :
                    : "r"(tmem_addr[0]), "l"(descA[0]), "l"(descB[0]), "r"(descI[0]), "r"(0), "r"(0), "r"(0), "r"(0), "r"(0)
                );
            }
        } else {
            
            {
                /* T.ptx_tcgen05_mma() */
                asm volatile(
                    "{\n"
                    ".reg .pred p;\n"
                    "setp.eq.u32 p, 1, 1; \n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "
                    "{%5, %6, %7, %8}, p;\n"
                    "}\n"
                    :
                    : "r"(tmem_addr[0]), "l"(descA[0]), "l"(descB[0]), "r"(descI[0]), "r"(1), "r"(0), "r"(0), "r"(0), "r"(0)
                );
            }
        }
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "l"(sem_addr) : "memory");
    }



    /* ---- wait for commit ---------------------------------------- */
    KDBG("waiting on mbarrier");
    mbarrier_wait<0>(&sem);
    KDBG("mbarrier passed");
    __syncthreads();

    /* ---- read a single column from TMEM ------------------------- */
    if (threadIdx.x == 0) KDBG("loading from TMEM");
    for (int i_1 = 0; i_1 < 128; ++i_1) {
        
        {
            /* T.ptx_tcgen05_ld() */
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x1.b32 "
                "{%0}, "
                "[%1];\n"
                :  "=r"(*(uint32_t*)&reg[i_1])
                :  "r"(get_tmem_addr(tmem_addr[0],  threadIdx.x /32 *32, i_1))
            );
        }
    }
    asm volatile(
                "tcgen05.wait::ld.sync.aligned;"
            );
    // for (int i_2 = 0; i_2 < 128; ++i_2) {
    //     C[((((int)threadIdx.x) * 128) + i_2)] = reg[i_2];
    // }
    int row = threadIdx.x;                       // 0 … 127 (row inside tile)

    int globalRow = row0 + row;                  // absolute row in C

    for (int n = 0; n < Nb; ++n) {               // Nb == 128
        int globalCol = col0 + n;                // absolute column in C
        int idx = globalRow * prob.N + globalCol;

        atomicAdd(&C[idx], reg[n]);              // merge partial sums
    }
    __syncthreads();
    KDBG("column written to global");
    /* ---- deallocate --------------------------------------------- */
    if (((int)threadIdx.x) < 32) {
        __asm__ __volatile__(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
      :: "r"(tmem_addr[0]), "r"(512)
      : "memory"
    );
  }
}

/* ---------------- CPU reference (full matrix) ------------------ */
void cpu_gemm(const std::vector<Tab>& A,
              const std::vector<Tab>& B,
              std::vector<Tacc>& C,
              ProblemSize prob)
{
    int M = prob.M, N = prob.N, K = prob.K;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int k = 0; k < K; ++k)
                s += __half2float(A[i*K + k]) *   // ←  K, not Kb
                      __half2float(B[j*K + k]);   // ←  K, not Kb
            C[i*N + j] = s;                       // ←  N, not Nb
        }
    }
}
/* =================================================================
 *                       B E N C H M A R K   D R I V E R
 * =================================================================*/

/* Forward declaration */
void gpu_run(const Tab* dA, const Tab* dB, Tacc* dC, ProblemSize  prob);

/* Lightning-style inner kernel launcher */
void gpu_run( Tab* dA,  Tab* dB, Tacc* dC, ProblemSize  prob)
{

    dim3 grid((prob.N + Nb - 1)/Nb,         // 8
            (prob.M + Mb - 1)/Mb,         // 8
            (prob.K + Kb - 1)/Kb);        // 16   split-K depth
    dim3 block(128);

    umma_tile<<<grid, block>>>(dA, dB, dC, prob);
}

/* Benchmark harness (matches ThunderKittens “run_benchmark”) */
int run_benchmark()
{
    

    std::mt19937 gen(41);
    std::uniform_real_distribution<float> rnd(-1.f, 1.f);
    ProblemSize prob{1024, 1024, 1024};
    int M = prob.M, N = prob.N, K = prob.K;   // convenience aliases
        // std::cout
        // << "--------------------  M=" << M
        // << " N=" << N << " K=" << K << "  --------------------\n"
        // << "Block size: " << Mb << "×" << Nb << " (single CTA)\n";


    /* ---------- host buffers ------------------------------------ */
    std::vector<float> hA_fp32(M * K);   // row-major A
    std::vector<float> hB_fp32(N * K);   // row-major B (transposed for GEMM)
    std::vector<Tab>   hA(M * K);
    std::vector<Tab>   hB(N * K);
    std::vector<Tacc>  hC_gpu(M * N, 0.f);
    std::vector<Tacc>  hC_ref(M * N, 0.f);

    /* fill A and B */
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            hA_fp32[i*K + j] = rnd(gen);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < K; ++j)
            hB_fp32[i*K + j] = rnd(gen);

    /* convert to half */
    for (size_t i = 0; i < hA.size(); ++i)  hA[i] = __float2half(hA_fp32[i]);
    for (size_t i = 0; i < hB.size(); ++i)  hB[i] = __float2half(hB_fp32[i]);

    /* ---------- device buffers ---------------------------------- */
    Tab  *dA, *dB;
    Tacc *dC;
    cudaMalloc(&dA, hA.size()*sizeof(Tab));   // M*K
    cudaMalloc(&dB, hB.size()*sizeof(Tab));   // N*K
    cudaMemcpy(dA, hA.data(), hA.size()*sizeof(Tab), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), hB.size()*sizeof(Tab), cudaMemcpyHostToDevice);
    cudaMalloc(&dC, hC_gpu.size()*sizeof(Tacc)); // M*N
    cudaMemset(dC, 0, hC_gpu.size()*sizeof(Tacc));

    std::cout << "Host data initialised\n";

    /* ---------- reference --------------------------------------- */
    cpu_gemm(hA, hB, hC_ref, prob);
    std::cout << "CPU reference done\n";

    /* ---------- device buffers ---------------------------------- */
    
    // cudaCheckErrors();
    std::cout << "device buffer done\n";

    /* ---------- timing ------------------------------------------ */
    const int iters = (std::getenv("NCU") ? 1 : 5);
    gpu_run(dA, dB, dC, prob);                       // 1 warm-up
    cudaDeviceSynchronize();
    std::cout << "warmup done\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        cudaMemset(dC, 0, M * N * sizeof(float));
        gpu_run(dA, dB, dC, prob); 
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double usec = std::chrono::duration<double>(t1 - t0).count()*1e6/iters;

    /* ---------- copy back & verify ------------------------------ */
    cudaMemcpy(hC_gpu.data(), dC, hC_gpu.size()*sizeof(Tacc),
               cudaMemcpyDeviceToHost);
    // cudaCheckErrors();
    std::cout << "copied back, doing error checking\n";

    /* ---------- full-matrix print + error stats -------------------- */
    double max_err = 0.0, avg_err = 0.0;          // pretty column width

    /* --- EXPECTED (CPU) ------------------------------------------- */
    // std::cout << "\n=== EXPECTED (CPU) ===\n     ";
    // for (int j = 0; j < N; ++j) std::cout << j;
    // std::cout << '\n';

    // for (int i = 0; i < M; ++i) {
    //     std::cout <<i << " :";
    //     for (int j = 0; j < N; ++j)
    //         std::cout <<  hC_ref[i * N + j];
    //     std::cout << '\n';
    // }

    /* --- GPU OUTPUT ------------------------------------------------ */
    // std::cout << "\n=== GPU OUTPUT ===\n     ";
    // for (int j = 0; j < N; ++j) std::cout <<  j;
    // std::cout << '\n';

    for (int i = 0; i < M; ++i) {
        //std::cout <<  i << " :";
        for (int j = 0; j < N; ++j) {
            float gpu = hC_gpu[i * N + j];
            //std::cout <<  gpu;

            /* accumulate error stats */
            double e = std::fabs(gpu - hC_ref[i * N + j]);
            max_err  = std::max(max_err, e);
            avg_err += e;
        }
        //std::cout << '\n';
    }
    avg_err /= static_cast<double>(M * N);
    std::cout << "\nSummary  →  max|err| = " << max_err
              << "   avg|err| = " << avg_err << '\n';
    double time_sec  = usec * 1e-6;              // average time per run
    double flops     = 2.0 * M * N * K;          // total ops per run
    double tflops    = flops / time_sec / 1e12;  // sustained TFLOPS

    std::cout << "Problem " << M << "x" << N << "x" << K
              << "  |  time = " << time_sec * 1e3 << " ms"
              << "  |  throughput = " << tflops << " TFLOPS\n";
    
    std::ofstream fout("matrix_dump.txt");
    std::streambuf* cout_buf = std::cout.rdbuf(); // backup
    std::cout.rdbuf(fout.rdbuf());                // redirect


    // std::cout << "\n=== EXPECTED (CPU) ===\n     ";
    // for (int j = 0; j < N; ++j) std::cout<< j;
    // std::cout << '\n';

    // for (int i = 0; i < M; ++i) {
    //     std::cout  << i << " :";
    //     for (int j = 0; j < N; ++j)
    //         std::cout << hC_ref[i * N + j];
    //     std::cout << '\n';
    // }

    // std::cout << "\n=== GPU OUTPUT ===\n     ";
    // for (int j = 0; j < N; ++j) std::cout << j;
    // std::cout << '\n';

    for (int i = 0; i < M; ++i) {
        //std::cout  << i << " :";
        for (int j = 0; j < N; ++j) {
            float gpu = hC_gpu[i * N + j];
            //std::cout << gpu;

            double e = std::fabs(gpu - hC_ref[i * N + j]);
            max_err  = std::max(max_err, e);
            avg_err += e;
        }
        //std::cout << '\n';
    }

    avg_err /= static_cast<double>(M * N);
    std::cout << "\nSummary  →  max|err| = " << max_err
              << "   avg|err| = " << avg_err << '\n';

    // std::cout.rdbuf(cout_buf); // restore std::cout
    /* ---------- clean up ---------------------------------------- */
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;

}

/* ------------------------- main ---------------------------------- */
int main()
{
    /* optional: force CUDA context creation upfront */
    cudaFree(nullptr);

    run_benchmark();
    return 0;
}