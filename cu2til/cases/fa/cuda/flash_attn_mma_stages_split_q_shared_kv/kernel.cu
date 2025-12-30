#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <algorithm>
#include <vector>
using namespace nvcuda;

#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CA(dst, src, bytes)                                     \
  asm volatile(                                                          \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
      "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                     \
  asm volatile(                                                          \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
      "l"(src), "n"(bytes))
#define CP_ASYNC_BULK_COMMIT_GROUP() \
  asm volatile("cp.async.bulk.commit_group;\n" ::)
#define CP_ASYNC_BULK_WAIT_ALL() asm volatile("cp.async.bulk.wait_all;\n" ::)
#define CP_ASYNC_BULK_WAIT_GROUP(n) \
  asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_BULK(dst, src, bytes)                                    \
  asm volatile(                                                           \
      "cp.async.bulk.global.shared::cta.bulk_group.L2::128B [%0], [%1], " \
      "%2;\n" ::"r"(dst),                                                 \
      "l"(src), "n"(bytes))
#define LDMATRIX_X1(R, addr)                                            \
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" \
               : "=r"(R)                                                \
               : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr)                                           \
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
               : "=r"(R0), "=r"(R1)                                         \
               : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                  \
  asm volatile(                                                            \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
      : "r"(addr))
#define LDMATRIX_X1_T(R, addr)                                                \
  asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" \
               : "=r"(R)                                                      \
               : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr)                                      \
  asm volatile(                                                          \
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" \
      : "=r"(R0), "=r"(R1)                                               \
      : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr)                               \
  asm volatile(                                                           \
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, " \
      "[%4];\n"                                                           \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                            \
      : "r"(addr))
#define STMATRIX_X1(addr, R)                                                \
  asm volatile(                                                             \
      "stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" ::"r"(addr), \
      "r"(R))
#define STMATRIX_X2(addr, R0, R1)                                         \
  asm volatile(                                                           \
      "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"( \
          addr),                                                          \
      "r"(R0), "r"(R1))
#define STMATRIX_X4(addr, R0, R1, R2, R3)                                     \
  asm volatile(                                                               \
      "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" :: \
          "r"(addr),                                                          \
      "r"(R0), "r"(R1), "r"(R2), "r"(R3))
#define STMATRIX_X1_T(addr, R)                                              \
  asm volatile(                                                             \
      "stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n" ::"r"( \
          addr),                                                            \
      "r"(R))
#define STMATRIX_X2_T(addr, R0, R1)                                         \
  asm volatile(                                                             \
      "stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n" :: \
          "r"(addr),                                                        \
      "r"(R0), "r"(R1))
#define STMATRIX_X4_T(addr, R0, R1, R2, R3)                                \
  asm volatile(                                                            \
      "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, " \
      "%4};\n" ::"r"(addr),                                                \
      "r"(R0), "r"(R1), "r"(R2), "r"(R3))
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)           \
  asm volatile(                                                               \
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, " \
      "%4, %5}, {%6, %7}, {%8, %9};\n"                                        \
      : "=r"(RD0), "=r"(RD1)                                                  \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), \
        "r"(RC1))
#define HMMA16816F32(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0,   \
                     RC1, RC2, RC3)                                           \
  asm volatile(                                                               \
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,  %1,  %2,  "    \
      "%3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"              \
      : "=r"(RD0), "=r"(RD1), "=r"(RD2), "=r"(RD3)                            \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), \
        "r"(RC1), "r"(RC2), "r"(RC3))

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                   \
  if (((T).options().dtype() != (th_type))) {                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                  \
  if (((T2).size(0) != (T1).size(0)) || ((T2).size(1) != (T1).size(1)) || \
      ((T2).size(2) != (T1).size(2)) || ((T2).size(3) != (T1).size(3))) { \
    throw std::runtime_error("Tensor size mismatch!");                    \
  }

HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

template <typename T, const int kWarpSize = WARP_SIZE>
DEVICE_INLINE T warp_reduce_sum(T val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, kWarpSize);
  }
  return val;
}

template <typename T, const int kWarpSize = WARP_SIZE>
DEVICE_INLINE T warp_reduce_max(T val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask, kWarpSize));
  }
  return val;
}

template <typename T, int M, const int N, const int K = 2>
DEVICE_INLINE void fill_3D_regs(T (&R)[M][N][K], T val) {
#pragma unroll
  for (int i = 0; i < M; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        R[i][j][k] = val;
      }
    }
  }
}

template <typename T, int M, const int N = 2>
DEVICE_INLINE void fill_2D_regs(T (&R)[M][N], T val) {
#pragma unroll
  for (int i = 0; i < M; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      R[i][j] = val;
    }
  }
}

template <typename T, int M>
DEVICE_INLINE void fill_1D_regs(T (&S)[M], T val) {
#pragma unroll
  for (int i = 0; i < M; ++i) {
    S[i] = val;
  }
}

template <typename T, int M>
DEVICE_INLINE void fill_1D_smem(T (&R)[M], T val, int tid) {
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < M; ++i) {
      R[i] = val;
    }
  }
}

#define INFHALF __float2half(65536.0f)
#define ZEROHALF __float2half(0.0f)

template <const int kHeadDim, const int kMmaAtomM, const int kMmaAtomN,
          const int kMmaAtomK, const int kMmaTileSeqLenQ,
          const int kMmaTileSeqLenK, const int kMmaTileSeqLenP,
          const int kMmaTileHeadDimV, const int kWarpTileSeqLenQ,
          const int kWarpTileSeqLenK, const int kWarpTileSeqLenP,
          const int kWarpTileHeadDimV, const int kOStorageAccFloat32,
          const int kStage, const int kPadQ, const int kPadK, const int kPadV>
__global__ void __launch_bounds__(WARP_SIZE *kMmaTileSeqLenQ *kMmaTileSeqLenK)
    flash_attn_mma_stages_split_q_shared_kv_kernel(half *Q, half *K, half *V,
                                                   half *O, int QKV_seqlen,
                                                   int QKV_head) {
  static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMmaAtomK == 16);
  static_assert(kMmaTileSeqLenQ <= 8 && kMmaTileSeqLenK == 1);
  static_assert(kMmaTileSeqLenP <= 8 && kMmaTileHeadDimV == 1);
  static_assert(kWarpTileSeqLenQ == 1 && kWarpTileSeqLenK <= 16);
  static_assert(kWarpTileSeqLenP == 1 &&
                kWarpTileHeadDimV ==
                    (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)));
  static_assert(kOStorageAccFloat32 == 0 || kOStorageAccFloat32 == 1);
  static_assert(kStage < 3 && kStage > 0);
  static_assert(kPadQ >= 0 && kPadQ % 8 == 0);
  static_assert(kPadK >= 0 && kPadK % 8 == 0);
  static_assert(kPadV >= 0 && kPadV % 8 == 0);
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  static_assert(Br >= Bc);
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  const int Tc = div_ceil(QKV_seqlen, Bc);
  const float scale = 1.0f / sqrt((float)kHeadDim);

  const int QKV_batch_id = blockIdx.y / QKV_head;
  const int QKV_head_id = blockIdx.y % QKV_head;
  const int Q_tile_id = blockIdx.x;
  const int O_tile_id = Q_tile_id;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_QP = warp_id;
  const int warp_KV = 0;
  const int Q_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
                             (QKV_head_id * QKV_seqlen * kHeadDim));
  const int K_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
                             (QKV_head_id * QKV_seqlen * kHeadDim));
  const int V_gmem_offset = Q_gmem_offset;
  const int O_gmem_offset = Q_gmem_offset;

  int load_smem_Q_Br = (tid / (kNumThreads / Br));
  int load_smem_Q_d =
      (tid % (kNumThreads / Br)) * (kHeadDim / (kNumThreads / Br));
  int load_smem_K_Bc = (tid / (kNumThreads / Bc));
  int load_smem_K_d =
      (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
  int load_smem_V_Bc = (tid / (kNumThreads / Bc));
  int load_smem_V_d =
      (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
  int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
  if (load_gmem_Q_Br >= QKV_seqlen) return;
  int load_gmem_K_Bc_offset = 0;
  int load_gmem_V_Bc_offset = 0;

  extern __shared__ half smem[];
  constexpr int Q_tile_size = Br * (kHeadDim + kPadQ);
  constexpr int K_tile_size = Bc * (kHeadDim + kPadK);
  constexpr int V_tile_size = Bc * (kHeadDim + kPadV);
  half *Q_tile_smem = smem;
  half *K_tile_smem = Q_tile_smem + Q_tile_size;
  half *V_tile_smem = K_tile_smem;

  uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  float lane_block_row_max_old[kWarpTileSeqLenQ][2];
  float lane_block_row_sum_old[kWarpTileSeqLenQ][2];
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

  constexpr bool kCanPrefetchQs2r =
      ((kHeadDim / kMmaAtomK) <= 8) && (kHeadDim < 64);
  constexpr bool kDelayPrefetchQs2r = (true && kCanPrefetchQs2r);
  constexpr bool kCanPrefetchKVg2s = (kStage == 2);
  constexpr int kPrefetchKg2sSmemId = 0;
  constexpr int kPrefetchVg2sSmemId = kCanPrefetchKVg2s ? 1 : 0;
  constexpr int kNumPrefetchQs2r =
      (kCanPrefetchQs2r) ? (kHeadDim / kMmaAtomK) : 1;
  uint32_t R_Q[kNumPrefetchQs2r][kWarpTileSeqLenQ][4];
  uint32_t R_K[kWarpTileSeqLenK][2];
  uint32_t R_V[kWarpTileHeadDimV][2];
  uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][2];
  uint32_t R_O[kWarpTileSeqLenP][kWarpTileHeadDimV][2];
  uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV]
              [(kOStorageAccFloat32) ? 4 : 2];
  fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV,
               ((kOStorageAccFloat32) ? 4 : 2)>(R_D, 0);

  {
    int load_gmem_Q_d = load_smem_Q_d;
    int load_gmem_Q_addr =
        (Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
    uint32_t load_smem_Q_ptr =
        (smem_Q_base_ptr +
         (load_smem_Q_Br * (kHeadDim + kPadQ) + load_smem_Q_d) * sizeof(half));
#pragma unroll
    for (int i = 0; i < (kHeadDim / (kNumThreads / Br)); i += 8) {
      CP_ASYNC_CG(load_smem_Q_ptr + i * 2, &Q[load_gmem_Q_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
  }

#pragma unroll 1
  for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; ++tile_K_seqlen) {
    if constexpr (kCanPrefetchKVg2s) {
      if (tile_K_seqlen == 0) {
        load_gmem_K_Bc_offset = tile_K_seqlen * Bc;
        int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
        int load_gmem_K_d = load_smem_K_d;
        int load_gmem_K_addr =
            (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
        uint32_t load_smem_K_ptr =
            (smem_K_base_ptr +
             (kPrefetchKg2sSmemId * K_tile_size +
              load_smem_K_Bc * (kHeadDim + kPadK) + load_smem_K_d) *
                 sizeof(half));
#pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
      }
      {
        load_gmem_V_Bc_offset = tile_K_seqlen * Bc;
        int load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
        int load_gmem_V_d = load_smem_V_d;
        int load_gmem_V_addr =
            (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
        uint32_t load_smem_V_ptr =
            (smem_V_base_ptr +
             (kPrefetchVg2sSmemId * V_tile_size +
              load_smem_V_Bc * (kHeadDim + kPadV) + load_smem_V_d) *
                 sizeof(half));
#pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }
    } else {
      load_gmem_K_Bc_offset = tile_K_seqlen * Bc;
      int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
      int load_gmem_K_d = load_smem_K_d;
      int load_gmem_K_addr =
          (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
      uint32_t load_smem_K_ptr =
          (smem_K_base_ptr +
           (kPrefetchKg2sSmemId * K_tile_size +
            load_smem_K_Bc * (kHeadDim + kPadK) + load_smem_K_d) *
               sizeof(half));
#pragma unroll
      for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
        CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
      }
      CP_ASYNC_COMMIT_GROUP();
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads();
    }

    if constexpr (kCanPrefetchQs2r && (!kDelayPrefetchQs2r)) {
      if (tile_K_seqlen == 0) {
        if constexpr (!kCanPrefetchKVg2s) {
          CP_ASYNC_WAIT_GROUP(0);
        } else {
          CP_ASYNC_WAIT_GROUP(1);
        }
        __syncthreads();

#pragma unroll
        for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
#pragma unroll
          for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
            int warp_smem_Q_Br =
                warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
            int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
            int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
            uint32_t lane_smem_Q_ptr =
                (smem_Q_base_ptr +
                 (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d) *
                     sizeof(half));
            LDMATRIX_X4(R_Q[tile_K_d][i][0], R_Q[tile_K_d][i][1],
                        R_Q[tile_K_d][i][2], R_Q[tile_K_d][i][3],
                        lane_smem_Q_ptr);
          }
        }
        __syncthreads();
      }
    }

    fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 2>(R_S, 0);
#pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      if constexpr (!kCanPrefetchQs2r) {
#pragma unroll
        for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
          int warp_smem_Q_Br =
              warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
          int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
          int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
          uint32_t lane_smem_Q_ptr =
              (smem_Q_base_ptr +
               (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d) *
                   sizeof(half));
          LDMATRIX_X4(R_Q[0][i][0], R_Q[0][i][1], R_Q[0][i][2], R_Q[0][i][3],
                      lane_smem_Q_ptr);
        }
      } else {
        if constexpr (kDelayPrefetchQs2r) {
          if (tile_K_seqlen == 0) {
            if (tile_K_d == 0) {
              if constexpr (!kCanPrefetchKVg2s) {
                CP_ASYNC_WAIT_GROUP(0);
              } else {
                CP_ASYNC_WAIT_GROUP(1);
              }
              __syncthreads();
            }
#pragma unroll
            for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
              int warp_smem_Q_Br =
                  warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
              int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
              int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
              uint32_t lane_smem_Q_ptr =
                  (smem_Q_base_ptr +
                   (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d) *
                       sizeof(half));
              LDMATRIX_X4(R_Q[tile_K_d][i][0], R_Q[tile_K_d][i][1],
                          R_Q[tile_K_d][i][2], R_Q[tile_K_d][i][3],
                          lane_smem_Q_ptr);
            }
          }
        }
      }

#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        int warp_smem_K_Bc =
            warp_KV * (kMmaAtomN * kWarpTileSeqLenK) + j * kMmaAtomN;
        int lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8;
        int lane_smem_K_d = tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8;
        uint32_t lane_smem_K_ptr =
            (smem_K_base_ptr +
             (kPrefetchKg2sSmemId * K_tile_size +
              lane_smem_K_Bc * (kHeadDim + kPadK) + lane_smem_K_d) *
                 sizeof(half));
        LDMATRIX_X2(R_K[j][0], R_K[j][1], lane_smem_K_ptr);
      }

      if constexpr (kCanPrefetchQs2r) {
        static_assert(kWarpTileSeqLenQ == 1);
        {
#pragma unroll
          for (int j = 0; j < kWarpTileSeqLenK; ++j) {
            HMMA16816(R_S[0][j][0], R_S[0][j][1], R_Q[tile_K_d][0][0],
                      R_Q[tile_K_d][0][1], R_Q[tile_K_d][0][2],
                      R_Q[tile_K_d][0][3], R_K[j][0], R_K[j][1], R_S[0][j][0],
                      R_S[0][j][1]);
          }
        }
      } else {
        static_assert(kWarpTileSeqLenQ == 1);
        {
#pragma unroll
          for (int j = 0; j < kWarpTileSeqLenK; ++j) {
            HMMA16816(R_S[0][j][0], R_S[0][j][1], R_Q[0][0][0], R_Q[0][0][1],
                      R_Q[0][0][2], R_Q[0][0][3], R_K[j][0], R_K[j][1],
                      R_S[0][j][0], R_S[0][j][1]);
          }
        }
      }
    }
    __syncthreads();

    if constexpr (!kCanPrefetchKVg2s) {
      load_gmem_V_Bc_offset = tile_K_seqlen * Bc;
      int load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
      int load_gmem_V_d = load_smem_V_d;
      int load_gmem_V_addr =
          (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
      uint32_t load_smem_V_ptr =
          (smem_V_base_ptr +
           (kPrefetchVg2sSmemId * V_tile_size +
            load_smem_V_Bc * (kHeadDim + kPadV) + load_smem_V_d) *
               sizeof(half));
#pragma unroll
      for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
        CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
      }
      CP_ASYNC_COMMIT_GROUP();
    }

    if constexpr (kCanPrefetchKVg2s) {
      if ((tile_K_seqlen + 1) < Tc) {
        load_gmem_K_Bc_offset = (tile_K_seqlen + 1) * Bc;
        int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
        int load_gmem_K_d = load_smem_K_d;
        int load_gmem_K_addr =
            (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
        uint32_t load_smem_K_ptr =
            (smem_K_base_ptr +
             (kPrefetchKg2sSmemId * K_tile_size +
              load_smem_K_Bc * (kHeadDim + kPadK) + load_smem_K_d) *
                 sizeof(half));
#pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }
    }

    float lane_row_max_new[kWarpTileSeqLenQ][2];
    float lane_row_sum_new[kWarpTileSeqLenQ][2];
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

    static_assert(kWarpTileSeqLenQ == 1);
    {
#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        half *t_hptr_S_0_1 = reinterpret_cast<half *>(&(R_S[0][j][0]));
        float tmp_max_0 =
            __half2float(__hmax(t_hptr_S_0_1[0], t_hptr_S_0_1[1])) * scale;
        float tmp_max_1 =
            __half2float(__hmax(t_hptr_S_0_1[2], t_hptr_S_0_1[3])) * scale;
        lane_row_max_new[0][0] = max(lane_row_max_new[0][0], tmp_max_0);
        lane_row_max_new[0][1] = max(lane_row_max_new[0][1], tmp_max_1);
      }

      lane_row_max_new[0][0] =
          warp_reduce_max<float, 4>(lane_row_max_new[0][0]);
      lane_row_max_new[0][1] =
          warp_reduce_max<float, 4>(lane_row_max_new[0][1]);
    }

    static_assert(kWarpTileSeqLenQ == 1);
    {
      float block_row_max_new_0 = lane_row_max_new[0][0];
      float block_row_max_new_1 = lane_row_max_new[0][1];

      float block_row_max_old_0 = lane_block_row_max_old[0][0];
      float block_row_max_old_1 = lane_block_row_max_old[0][1];
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        half *t_hptr_S_0_1 = reinterpret_cast<half *>(&(R_S[0][j][0]));
        float4 t_reg_S_0_1;
        t_reg_S_0_1.x = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[0]), scale,
                                         -block_row_max_new_0));
        t_reg_S_0_1.y = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[1]), scale,
                                         -block_row_max_new_0));
        t_reg_S_0_1.z = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[2]), scale,
                                         -block_row_max_new_1));
        t_reg_S_0_1.w = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[3]), scale,
                                         -block_row_max_new_1));
        lane_row_sum_new[0][0] += (t_reg_S_0_1.x + t_reg_S_0_1.y);
        lane_row_sum_new[0][1] += (t_reg_S_0_1.z + t_reg_S_0_1.w);
        t_hptr_S_0_1[0] = __float2half_rn(t_reg_S_0_1.x);
        t_hptr_S_0_1[1] = __float2half_rn(t_reg_S_0_1.y);
        t_hptr_S_0_1[2] = __float2half_rn(t_reg_S_0_1.z);
        t_hptr_S_0_1[3] = __float2half_rn(t_reg_S_0_1.w);
      }

      lane_row_sum_new[0][0] =
          warp_reduce_sum<float, 4>(lane_row_sum_new[0][0]);
      lane_row_sum_new[0][1] =
          warp_reduce_sum<float, 4>(lane_row_sum_new[0][1]);
    }

    if constexpr (kCanPrefetchKVg2s) {
      if ((tile_K_seqlen + 1) < Tc) {
        CP_ASYNC_WAIT_GROUP(1);
      } else {
        CP_ASYNC_WAIT_GROUP(0);
      }
    } else {
      CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, 2>(R_O, 0);
#pragma unroll
    for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
#pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        int warp_smem_V_d =
            warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
        int lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16;
        int lane_smem_V_d = warp_smem_V_d;
        uint32_t lane_smem_V_ptr =
            (smem_V_base_ptr +
             (kPrefetchVg2sSmemId * V_tile_size +
              lane_smem_V_Bc * (kHeadDim + kPadV) + lane_smem_V_d) *
                 sizeof(half));
        LDMATRIX_X2_T(R_V[j][0], R_V[j][1], lane_smem_V_ptr);
      }

      int w = tile_V_Bc * 2;
      static_assert(kWarpTileSeqLenP == 1);
      {
#pragma unroll
        for (int j = 0; j < kWarpTileHeadDimV; ++j) {
          HMMA16816(R_O[0][j][0], R_O[0][j][1], R_S[0][w][0], R_S[0][w][1],
                    R_S[0][w + 1][0], R_S[0][w + 1][1], R_V[j][0], R_V[j][1],
                    R_O[0][j][0], R_O[0][j][1]);
        }
      }
    }
    __syncthreads();

    static_assert(kWarpTileSeqLenP == 1);
    {
      float block_row_max_new_0 = lane_row_max_new[0][0];
      float block_row_max_new_1 = lane_row_max_new[0][1];
      float block_row_sum_new_0 = lane_row_sum_new[0][0];
      float block_row_sum_new_1 = lane_row_sum_new[0][1];

      float block_row_max_old_0 = lane_block_row_max_old[0][0];
      float block_row_max_old_1 = lane_block_row_max_old[0][1];
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);
      block_row_max_old_0 =
          (tile_K_seqlen > 0 ? block_row_max_old_0 : block_row_max_new_0);
      block_row_max_old_1 =
          (tile_K_seqlen > 0 ? block_row_max_old_1 : block_row_max_new_1);

      float rescale_o_factor_0 =
          __expf(block_row_max_old_0 - block_row_max_new_0);
      float rescale_o_factor_1 =
          __expf(block_row_max_old_1 - block_row_max_new_1);
#pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        half *t_hptr_O_0_1 = reinterpret_cast<half *>(&(R_O[0][j][0]));
        if constexpr (kOStorageAccFloat32) {
          float *t_fptr_D_0_1 = reinterpret_cast<float *>(&(R_D[0][j][0]));
          t_fptr_D_0_1[0] = __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[0],
                                      __half2float(t_hptr_O_0_1[0]));
          t_fptr_D_0_1[1] = __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[1],
                                      __half2float(t_hptr_O_0_1[1]));
          t_fptr_D_0_1[2] = __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[2],
                                      __half2float(t_hptr_O_0_1[2]));
          t_fptr_D_0_1[3] = __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[3],
                                      __half2float(t_hptr_O_0_1[3]));
        } else {
          half *t_hptr_D_0_1 = reinterpret_cast<half *>(&(R_D[0][j][0]));
          t_hptr_D_0_1[0] = __float2half_rn(
              __fmaf_rn(rescale_o_factor_0, __half2float(t_hptr_D_0_1[0]),
                        __half2float(t_hptr_O_0_1[0])));
          t_hptr_D_0_1[1] = __float2half_rn(
              __fmaf_rn(rescale_o_factor_0, __half2float(t_hptr_D_0_1[1]),
                        __half2float(t_hptr_O_0_1[1])));
          t_hptr_D_0_1[2] = __float2half_rn(
              __fmaf_rn(rescale_o_factor_1, __half2float(t_hptr_D_0_1[2]),
                        __half2float(t_hptr_O_0_1[2])));
          t_hptr_D_0_1[3] = __float2half_rn(
              __fmaf_rn(rescale_o_factor_1, __half2float(t_hptr_D_0_1[3]),
                        __half2float(t_hptr_O_0_1[3])));
        }
      }

      float block_row_sum_old_0 = lane_block_row_sum_old[0][0];
      float block_row_sum_old_1 = lane_block_row_sum_old[0][1];
      lane_block_row_sum_old[0][0] = (__fmaf_rn(
          rescale_o_factor_0, block_row_sum_old_0, block_row_sum_new_0));
      lane_block_row_sum_old[0][1] = (__fmaf_rn(
          rescale_o_factor_1, block_row_sum_old_1, block_row_sum_new_1));
      lane_block_row_max_old[0][0] = block_row_max_new_0;
      lane_block_row_max_old[0][1] = block_row_max_new_1;
    }

    if constexpr (kCanPrefetchKVg2s) {
      if ((tile_K_seqlen + 1) < Tc) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
      }
    }
  }
  __syncthreads();

  static_assert(kWarpTileSeqLenP == 1);
  {
    float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[0][0]);
    float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[0][1]);
#pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {
      if constexpr (kOStorageAccFloat32) {
        float *t_fptr_D_0_1 = reinterpret_cast<float *>(&(R_D[0][j][0]));
        half *t_hptr_D_0_1 = reinterpret_cast<half *>(&(R_D[0][j][0]));
        t_hptr_D_0_1[0] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[0]);
        t_hptr_D_0_1[1] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[1]);
        t_hptr_D_0_1[2] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[2]);
        t_hptr_D_0_1[3] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[3]);
      } else {
        half *t_hptr_D_0_1 = reinterpret_cast<half *>(&(R_D[0][j][0]));
        t_hptr_D_0_1[0] =
            __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D_0_1[0]));
        t_hptr_D_0_1[1] =
            __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D_0_1[1]));
        t_hptr_D_0_1[2] =
            __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D_0_1[2]));
        t_hptr_D_0_1[3] =
            __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D_0_1[3]));
      }
    }
  }

  static_assert(kWarpTileSeqLenP == 1);
  {
#pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {
      if constexpr (kCanPrefetchQs2r && kNumPrefetchQs2r > 1) {
        R_Q[0][0][0] = R_D[0][j][0];
        R_Q[1][0][0] = R_D[0][j][1];
        R_Q[0][0][1] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 1, 4);
        R_Q[0][0][2] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 2, 4);
        R_Q[0][0][3] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 3, 4);
        R_Q[1][0][1] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 1, 4);
        R_Q[1][0][2] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 2, 4);
        R_Q[1][0][3] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 3, 4);
        if (lane_id % 4 == 0) {
          int store_warp_regs_O_Br =
              warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + 0 * kMmaAtomM;
          int store_lane_gmem_O_Br =
              O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
          int store_warp_regs_O_d =
              warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
          int store_lane_gmem_O_d = store_warp_regs_O_d;
          int store_gmem_O_addr_0 =
              (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim +
               store_lane_gmem_O_d);
          int store_gmem_O_addr_1 =
              (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim +
               store_lane_gmem_O_d);
          LDST128BITS(O[store_gmem_O_addr_0]) = LDST128BITS(R_Q[0][0][0]);
          LDST128BITS(O[store_gmem_O_addr_1]) = LDST128BITS(R_Q[1][0][0]);
        }
      } else {
        uint32_t R_Z[2][4];
        R_Z[0][0] = R_D[0][j][0];
        R_Z[1][0] = R_D[0][j][1];
        R_Z[0][1] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 1, 4);
        R_Z[0][2] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 2, 4);
        R_Z[0][3] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 3, 4);
        R_Z[1][1] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 1, 4);
        R_Z[1][2] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 2, 4);
        R_Z[1][3] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 3, 4);
        if (lane_id % 4 == 0) {
          int store_warp_regs_O_Br =
              warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + 0 * kMmaAtomM;
          int store_lane_gmem_O_Br =
              O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
          int store_warp_regs_O_d =
              warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
          int store_lane_gmem_O_d = store_warp_regs_O_d;
          int store_gmem_O_addr_0 =
              (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim +
               store_lane_gmem_O_d);
          int store_gmem_O_addr_1 =
              (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim +
               store_lane_gmem_O_d);
          LDST128BITS(O[store_gmem_O_addr_0]) = LDST128BITS(R_Z[0][0]);
          LDST128BITS(O[store_gmem_O_addr_1]) = LDST128BITS(R_Z[1][0]);
        }
      }
    }
  }
}

template <const int kHeadDim, const int kStage>
void launch_flash_attn_mma_stages_split_q_shared_kv(torch::Tensor Q,
                                                    torch::Tensor K,
                                                    torch::Tensor V,
                                                    torch::Tensor O) {
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenQ = 4;
  constexpr int kMmaTileSeqLenK = 1;
  constexpr int kMmaTileSeqLenP = 4;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  constexpr int kWarpTileSeqLenK = (kStage > 1) ? 4 : 8;
  constexpr int kWarpTileSeqLenP = 1;
  constexpr int kWarpTileHeadDimV = (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV));
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  constexpr int kPadQ = 8;
  constexpr int kPadK = 8;
  constexpr int kPadV = 8;
  constexpr int kOStorageAccFloat32 = (kHeadDim < 256) ? 1 : 0;

  constexpr int Q_tile_size = (Br * (kHeadDim + kPadQ));
  constexpr int K_tile_size = (Bc * (kHeadDim + kPadK));
  constexpr int V_tile_size = (Bc * (kHeadDim + kPadV));
  const int smem_max_size =
      (Q_tile_size + kStage * max(K_tile_size, V_tile_size)) * sizeof(half);

  const int QKV_batch = Q.size(0);
  const int QKV_head = Q.size(1);
  const int QKV_seqlen = Q.size(2);
  assert(QKV_seqlen % max(Br, Bc) == 0);

  dim3 grid(div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head);
  dim3 block(kNumThreads);

  cudaFuncSetAttribute(
      flash_attn_mma_stages_split_q_shared_kv_kernel<
          kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
          kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
          kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
          kOStorageAccFloat32, kStage, kPadQ, kPadK, kPadV>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

  flash_attn_mma_stages_split_q_shared_kv_kernel<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
      kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
      kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
      kOStorageAccFloat32, kStage, kPadQ, kPadK, kPadV>
      <<<grid, block, smem_max_size>>>(reinterpret_cast<half *>(Q.data_ptr()),
                                       reinterpret_cast<half *>(K.data_ptr()),
                                       reinterpret_cast<half *>(V.data_ptr()),
                                       reinterpret_cast<half *>(O.data_ptr()),
                                       QKV_seqlen, QKV_head);
}

void flash_attn_mma_stages_split_q_shared_kv(torch::Tensor Q, torch::Tensor K,
                                             torch::Tensor V, torch::Tensor O) {
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf)
  const int d = Q.size(3);

  switch (d) {
    case 32:
      launch_flash_attn_mma_stages_split_q_shared_kv<32, 2>(Q, K, V, O);
      break;
    case 64:
      launch_flash_attn_mma_stages_split_q_shared_kv<64, 2>(Q, K, V, O);
      break;
    case 96:
      launch_flash_attn_mma_stages_split_q_shared_kv<96, 2>(Q, K, V, O);
      break;
    case 128:
      launch_flash_attn_mma_stages_split_q_shared_kv<128, 2>(Q, K, V, O);
      break;
    default:
      throw std::runtime_error("headdim not support!");
      break;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(flash_attn_mma_stages_split_q_shared_kv)
}