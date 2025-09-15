#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <vector>

using namespace nvcuda;

// --- Preserved Macros as per instructions ---
#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                  \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                         \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }
// --- End of Preserved Macros ---


HOST_DEVICE_INLINE
int ceil_div(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = true, const bool COLLECTIVE_STORE = false>
__global__ void __launch_bounds__(256)
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel(half *A, half *B,
                                                         half *C, int M, int N,
                                                         int K) {
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = ceil_div(K, MMA_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;
  constexpr int BK = MMA_K;

  extern __shared__ half smem[];
  half *s_a = smem;
  half *s_b = smem + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_m = warp_id % 2;
  const int warp_n = warp_id / 2;

  int load_smem_a_m = tid / 2;
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;
  int load_smem_b_k = tid / 16;
  int load_smem_b_n = (tid % 16) * 8;
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }

  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) {
    int load_gmem_a_k = k * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
             sizeof(half));
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(
            load_smem_a_ptr),
        "l"(&A[load_gmem_a_addr]), "n"(16));

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr +
         (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
             sizeof(half));
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(
            load_smem_b_ptr),
        "l"(&B[load_gmem_b_addr]), "n"(16));

    asm volatile("cp.async.commit_group;\n" ::);
  }

  asm volatile("cp.async.wait_group %0;\n" ::"n"(K_STAGE - 2));
  __syncthreads();

#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    int smem_sel = (k + 1) % K_STAGE;
    int smem_sel_next = k % K_STAGE;

    int load_gmem_a_k = k * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset +
                            load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
                               sizeof(half));
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(
            load_smem_a_ptr),
        "l"(&A[load_gmem_a_addr]), "n"(16));

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset +
                            load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
                               sizeof(half));
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(
            load_smem_b_ptr),
        "l"(&B[load_gmem_b_addr]), "n"(16));
    asm volatile("cp.async.commit_group;\n" ::);

    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
      int lane_smem_a_k = (lane_id / 16) * 8;
      uint32_t lane_smem_a_ptr =
          (smem_a_base_ptr + (smem_sel * s_a_stage_offset +
                              lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) *
                                 sizeof(half));
      asm volatile(
          "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
          : "=r"(RA[i][0]), "=r"(RA[i][1]), "=r"(RA[i][2]), "=r"(RA[i][3])
          : "r"(lane_smem_a_ptr));
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;
      int lane_smem_b_n = warp_smem_b_n;
      uint32_t lane_smem_b_ptr =
          (smem_b_base_ptr + (smem_sel * s_b_stage_offset +
                              lane_smem_b_k * (BN + B_PAD) + lane_smem_b_n) *
                                 sizeof(half));
      asm volatile(
          "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
          : "=r"(RB[j][0]), "=r"(RB[j][1])
          : "r"(lane_smem_b_ptr));
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, "
            "%3, "
            "%4, %5}, {%6, %7}, {%8, %9};\n"
            : "=r"(RC[i][j][0]), "=r"(RC[i][j][1])
            : "r"(RA[i][0]), "r"(RA[i][1]), "r"(RA[i][2]), "r"(RA[i][3]),
              "r"(RB[j][0]), "r"(RB[j][1]), "r"(RC[i][j][0]), "r"(RC[i][j][1]));
      }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(K_STAGE - 2));
    __syncthreads();
  }

  if constexpr ((K_STAGE - 2) > 0) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    __syncthreads();
  }

  {
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      uint32_t RA[WARP_TILE_M][4];
      uint32_t RB[WARP_TILE_N][2];

      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
        int lane_smem_a_k = (lane_id / 16) * 8;
        uint32_t lane_smem_a_ptr =
            (smem_a_base_ptr + (stage_sel * s_a_stage_offset +
                                lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) *
                                   sizeof(half));
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(RA[i][0]), "=r"(RA[i][1]), "=r"(RA[i][2]), "=r"(RA[i][3])
            : "r"(lane_smem_a_ptr));
      }

#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16;
        int lane_smem_b_n = warp_smem_b_n;
        uint32_t lane_smem_b_ptr =
            (smem_b_base_ptr + (stage_sel * s_b_stage_offset +
                                lane_smem_b_k * (BN + B_PAD) + lane_smem_b_n) *
                                   sizeof(half));
        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(RB[j][0]), "=r"(RB[j][1])
            : "r"(lane_smem_b_ptr));
      }

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, "
              "%3, "
              "%4, %5}, {%6, %7}, {%8, %9};\n"
              : "=r"(RC[i][j][0]), "=r"(RC[i][j][1])
              : "r"(RA[i][0]), "r"(RA[i][1]), "r"(RA[i][2]), "r"(RA[i][3]),
                "r"(RB[j][0]), "r"(RB[j][1]), "r"(RC[i][j][0]),
                "r"(RC[i][j][1]));
        }
      }
    }
  }

  {
    for (int i = 0; i < WARP_TILE_M; ++i) {
      uint32_t RC0[WARP_TILE_N][4];
      uint32_t RC1[WARP_TILE_N][4];
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        RC0[j][0] = RC[i][j][0];
        RC1[j][0] = RC[i][j][1];
        RC0[j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
        RC0[j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
        RC0[j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
        RC1[j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
        RC1[j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
        RC1[j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
      }

      if (lane_id % 4 == 0) {
        int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
          int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n;
          int store_gmem_c_addr_0 =
              store_lane_gmem_c_m * N + store_lane_gmem_c_n;
          int store_gmem_c_addr_1 =
              (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
          (reinterpret_cast<float4 *>(&(C[store_gmem_c_addr_0]))[0]) = ((reinterpret_cast<float4 *>(&(RC0[j][0]))[0]));
          (reinterpret_cast<float4 *>(&(C[store_gmem_c_addr_1]))[0]) = ((reinterpret_cast<float4 *>(&(RC1[j][0]))[0]));
        }
      }
    }
  }
}

extern "C" void cuda_kernel(half* A, half* B, half* C, int M, int K, int N) {
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int MMA_TILE_M = 2;
  constexpr int MMA_TILE_N = 4;
  constexpr int WARP_TILE_M = 4;
  constexpr int WARP_TILE_N = 4;
  constexpr int A_PAD = 8;
  constexpr int B_PAD = 8;
  constexpr int NUM_THREADS = (MMA_TILE_M * MMA_TILE_N * WARP_SIZE);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;
  constexpr int BK = MMA_K;

  float swizzle_factor = (N <= 4096) ? 0.5f : 0.25f;
  if (N >= 14848 && K > 8192 && N % 8 == 0) {
    swizzle_factor = 0.125f;
  }

  int swizzle_stride = static_cast<int>(N * swizzle_factor);
  swizzle_stride = (swizzle_stride >= 256) ? swizzle_stride : 1;

  const int smem_max_size = ((3) * BM * (BK + A_PAD) * sizeof(half) +
                             (3) * BK * (BN + B_PAD) * sizeof(half));
  cudaFuncSetAttribute(hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel<
                           MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,
                           WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (3), true>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
  const int N_SWIZZLE = (N + (swizzle_stride)-1) / (swizzle_stride);
  dim3 block(NUM_THREADS);
  dim3 grid((ceil_div(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, ceil_div(M, BM),
            N_SWIZZLE);
  hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel<
      MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M, WARP_TILE_N,
      A_PAD, B_PAD, (3), true><<<grid, block, smem_max_size>>>(
      A, B, C, M, N, K);
}
