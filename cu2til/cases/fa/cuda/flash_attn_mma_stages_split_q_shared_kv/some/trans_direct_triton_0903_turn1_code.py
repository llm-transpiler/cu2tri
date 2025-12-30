from google import genai
from google.genai import types
import base64

def generate():
  client = genai.Client(
      vertexai=True,
      project="totemic-fact-469314-n2",
      location="global",
  )

  msg1_text1 = types.Part.from_text(text="""#include <cuda_bf16.h>#include <cuda_fp16.h>#include <cuda_fp8.h>#include <cuda_runtime.h>#include <float.h>#include <mma.h>#include <stdint.h>#include <stdio.h>#include <stdlib.h>#include <torch/extension.h>#include <torch/types.h>
#include <algorithm>#include <vector>using namespace nvcuda;
#define WARP_SIZE 32
template <const int kHeadDim, const int kMmaAtomM = 16, const int kMmaAtomN = 8,
          const int kMmaAtomK = 16, const int kMmaTileSeqLenQ = 4,
          const int kMmaTileSeqLenK = 1, const int kMmaTileSeqLenP = 4,
          const int kMmaTileHeadDimV = 1, const int kWarpTileSeqLenQ = 1,
          const int kWarpTileSeqLenK = 4, const int kWarpTileSeqLenP = 1,
          const int kWarpTileHeadDimV = (kHeadDim / 8),
          const int kStage = 2>__global__ void __launch_bounds__(WARP_SIZE *kMmaTileSeqLenQ *kMmaTileSeqLenK)
    flash_attn_mma_stages_split_q_shared_kv_kernel(half *Q, half *K, half *V,
                                                   half *O, int QKV_seqlen,
                                                   int QKV_head, float scale) {  // kHeadDim is just 
  static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMmaAtomK == 16);
  static_assert(kMmaTileSeqLenQ <= 8 && kMmaTileSeqLenK == 1);
  static_assert(kMmaTileSeqLenP <= 8 && kMmaTileHeadDimV == 1);
  static_assert(kWarpTileSeqLenQ == 1 && kWarpTileSeqLenK <= 16);
  static_assert(kWarpTileSeqLenP == 1 &&
                kWarpTileHeadDimV ==
                    (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)));
  static_assert(kStage == 2);
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // 16 * 4 * 1 = 64
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; // 8 * 1 * 4 = 32
  static_assert(Br >= Bc);
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 32 * 4 * 1 = 128
  assert(QKV_seqlen % Bc == 0);
  const int Tc = QKV_seqlen / Bc;

  const int QKV_batch_id = blockIdx.y / QKV_head;
  const int QKV_head_id = blockIdx.y % QKV_head;
  const int Q_tile_id = blockIdx.x;
  const int O_tile_id = Q_tile_id;
  const int tid = threadIdx.x; // 0...127
  const int warp_id = tid / WARP_SIZE; // 0...4
  const int lane_id = tid % WARP_SIZE; // 0...31
  const int warp_QP = warp_id;
  const int warp_KV = 0;
  const int Q_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
                             (QKV_head_id * QKV_seqlen * kHeadDim));   // make_coord_style_offset(coord=(batch_id, head_id), stride=(QKV_head, 1), step_tile=(QKV_seqlen, kHeadDim))
  const int K_gmem_offset = Q_gmem_offset;
  const int V_gmem_offset = Q_gmem_offset;
  const int O_gmem_offset = Q_gmem_offset;
  // (kNumThreads / Br) = (128 / 64) = 2: (Br, kHeadDim)的smem块每行有 2 个线程处理
  int load_smem_Q_Br = (tid / (kNumThreads / Br)); // 标记当前tid在哪个Br块中, tid:(0,...,kNumThreads-1)
  int load_smem_Q_d = (tid % (kNumThreads / Br)) * (kHeadDim / (kNumThreads / Br)); // (tid % 2) * (d / 2)  // coord of smem of q tid -> (load_smem_Q_Br, load_smem_Q_d) => (tid / 2, (tid % 2) * col/2 )  // 可以让llm尝试推断模式, 失败则保留原代码  // load_smem_Q_Br: block_tile(Br, kHeadDim)的行id(每个线程都对应一个行id)  // load_smem_Q_d: block_tile(Br, kHeadDim)的列id(每个线程都对应一个列id)  // 这是一个(Br, 128/Br):(128/Br, 1)=>(Br, 2):(2, 1) -> (1, d/2) of Tile(Br, kHeadDim)的映射
  int load_smem_K_Bc = (tid / (kNumThreads / Bc));
  int load_smem_K_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
  int load_smem_V_Bc = (tid / (kNumThreads / Bc));
  int load_smem_V_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
  int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;  // tid 需要去gmem上load的 行坐标
  if (load_gmem_Q_Br >= QKV_seqlen) return;
  int load_gmem_K_Bc_offset = 0;  // int load_gmem_V_Bc_offset = 0;

  extern __shared__ half smem[];
  constexpr int Q_tile_size = Br * kHeadDim;  // constexpr int K_tile_size = Bc * kHeadDim;
  constexpr int V_tile_size = Bc * kHeadDim;
  half *Q_tile_smem = smem;
  half *K_tile_smem = Q_tile_smem + Q_tile_size;
  half *V_tile_smem = K_tile_smem;

  uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  float lane_block_row_max_old[kWarpTileSeqLenQ][2]; // [1][2]
  float lane_block_row_sum_old[kWarpTileSeqLenQ][2]; // [1][2]
  {#pragma unroll
    for (int i = 0; i < kWarpTileSeqLenQ; ++i) {#pragma unroll
      for (int j = 0; j < 2; ++j) {
        lane_block_row_max_old[i][j] = -INFINITY;
      }
    }
  }
  {#pragma unroll
    for (int i = 0; i < kWarpTileSeqLenQ; ++i) {#pragma unroll
      for (int j = 0; j < 2; ++j) {
        lane_block_row_sum_old[i][j] = 0.0f;
      }
    }
  }

  uint32_t R_Q[kWarpTileSeqLenQ][4]; // [1][4]
  uint32_t R_K[kWarpTileSeqLenK][2]; // [4][2]
  uint32_t R_V[kWarpTileHeadDimV][2]; // [(kHeadDim / 8)][2]
  uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][2]; // [1][4][2]
  uint32_t R_O[kWarpTileSeqLenP][kWarpTileHeadDimV][2]; // [1][(kHeadDim / 8)][2]
  uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV][4]; // [1][(kHeadDim / 8)][4]
  {#pragma unroll
    for (int i = 0; i < kWarpTileSeqLenP; ++i) {#pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {#pragma unroll
        for (int k = 0; k < 4; ++k) {
          R_D[i][j][k] = 0;
        }
      }
    }
  }

  {
    int load_gmem_Q_d = load_smem_Q_d;
    int load_gmem_Q_addr =
        (Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);    // gmem_coord => (load_gmem_Q_Br, load_gmem_Q_d):(kHeadDim, 1)
    uint32_t load_smem_Q_ptr =
        (smem_Q_base_ptr +
         (load_smem_Q_Br * kHeadDim + load_smem_Q_d) * sizeof(half));    // smem_coord => (load_smem_Q_Br, load_smem_Q_d):(kHeadDim, 1)#pragma unroll
    for (int i = 0; i < (kHeadDim / (kNumThreads / Br)); i += 8) {
      asm volatile( 
      \"cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\
\" ::\"r\"(load_smem_Q_ptr + i * 2), 
      \"l\"(&Q[load_gmem_Q_addr + i]), \"n\"(16)); // 传输16字节 = 8个half元素
    }
    asm volatile(\"cp.async.commit_group;\
\" ::);
  }
#pragma unroll 1
  for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; ++tile_K_seqlen) {
    if (tile_K_seqlen == 0) {      /*      * copy G2S      * copy_shape: copy_block_tile_shape: (Bc, kHeadDim)      * block_coord: (tile_K_seqlen, 0); step_tile:(Bc, kHeadDim)      * thread_copy_coord, in block_tile:       *   smem_coord: (load_smem_K_Bc, load_smem_K_d):(kHeadDim, 1):(0,0):(Bc, kHeadDim):smem_K_base_ptr      *   gmem_coord: (load_gmem_K_Bc, load_gmem_K_d)=(load_smem_K_Bc, load_smem_K_d):(kHeadDim, 1):(tile_K_seqlen * Bc, 0):(QKV_seqlen, kHeadDim):&K[K_gmem_offset]      * (crd_dim1, crd_dim2):(crd_stride_dim1, crd_stride_dim2):(start_offset_dim1, start_offset_dim2):(crd_space_dim1, crd_space_dim2)      */
      int load_gmem_K_Bc = tile_K_seqlen * Bc + load_smem_K_Bc;
      int load_gmem_K_d = load_smem_K_d;
      int load_gmem_K_addr = K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d;
      uint32_t load_smem_K_ptr = smem_K_base_ptr + (load_smem_K_Bc * kHeadDim + load_smem_K_d) * sizeof(half); // 每个线程拷贝的目标smem地址#pragma unroll
      for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
        asm volatile( 
            \"cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\
\" ::\"r\"(load_smem_K_ptr + i * 2), 
            \"l\"(&K[load_gmem_K_addr + i]), \"n\"(16));
      }
      asm volatile(\"cp.async.commit_group;\
\" ::);

      asm volatile(\"cp.async.wait_group %0;\
\" ::\"n\"(0));
      __syncthreads();
    }
    {      /*      * copy G2S      * copy_shape: copy_block_tile_shape: (Bc, kHeadDim)      * block_coord: (tile_K_seqlen, 0); step_tile:(Bc, kHeadDim)      * thread_copy_coord, in block_tile:       *   smem_coord: (load_smem_V_Bc, load_smem_V_d):(kHeadDim, 1):(0,0):(Bc, kHeadDim):smem_V_base_ptr + V_tile_size * sizeof(half)      *   gmem_coord: (load_gmem_V_Bc, load_gmem_V_d)=(load_smem_V_Bc, load_smem_V_d):(kHeadDim, 1):(tile_K_seqlen * Bc, 0):(QKV_seqlen, kHeadDim):&V[V_gmem_offset]      * (crd_dim1, crd_dim2):(crd_stride_dim1, crd_stride_dim2):(start_offset_dim1, start_offset_dim2):(crd_space_dim1, crd_space_dim2)      */
      int load_gmem_V_Bc = tile_K_seqlen * Bc + load_smem_V_Bc; // gmem_coord_dim1
      int load_gmem_V_d = load_smem_V_d; // gmem_coord_dim2
      int load_gmem_V_addr = V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d; // gmem_addr
      uint32_t load_smem_V_ptr = smem_V_base_ptr + (V_tile_size + load_smem_V_Bc * kHeadDim + load_smem_V_d) * sizeof(half); // smem_addr#pragma unroll
      for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
        asm volatile( 
            \"cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\
\" ::\"r\"(load_smem_V_ptr + i * 2), 
            \"l\"(&V[load_gmem_V_addr + i]), \"n\"(16));
      }
      asm volatile(\"cp.async.commit_group;\
\" ::);
    }
    {#pragma unroll
      for (int i = 0; i < kWarpTileSeqLenQ; ++i) { // 1#pragma unroll
        for (int j = 0; j < kWarpTileSeqLenK; ++j) { // 4#pragma unroll
          for (int k = 0; k < 2; ++k) {
            R_S[i][j][k] = 0;
          }
        }
      }
    }#pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) { // d / 16#pragma unroll
      for (int i = 0; i < kWarpTileSeqLenQ; ++i) { // 1
        int warp_smem_Q_Br = warp_id * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM; // warp在smem上的行坐标起始
        int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16; // lane在warp中的smem行坐标, lane本身排布列优先, 所以%16, 符合ldmatrix标准
        int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8; // lane在warp中的smem列坐标
        uint32_t lane_smem_Q_ptr =
            (smem_Q_base_ptr +
             (lane_smem_Q_Br * kHeadDim + lane_smem_Q_d) *
                 sizeof(half));
        asm volatile( 
            \"ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\
\" 
            : \"=r\"(R_Q[i][0]), \"=r\"(R_Q[i][1]), \"=r\"(R_Q[i][2]), \"=r\"(R_Q[i][3]) 
            : \"r\"(lane_smem_Q_ptr));
      }
#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) { // 4
        int warp_smem_K_Bc =
            warp_KV * (kMmaAtomN * kWarpTileSeqLenK) + j * kMmaAtomN; // warp_KV == 0
        int lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8;
        int lane_smem_K_d = tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8; // 0...7 -> 0; 8...15 -> 8
        uint32_t lane_smem_K_ptr =
            (smem_K_base_ptr + (lane_smem_K_Bc * kHeadDim + lane_smem_K_d) * sizeof(half));
        asm volatile(\"ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\
\" 
                   : \"=r\"(R_K[j][0]), \"=r\"(R_K[j][1]) 
                   : \"r\"(lane_smem_K_ptr));
      }

      {#pragma unroll
        for (int j = 0; j < kWarpTileSeqLenK; ++j) { // 4
          asm volatile( 
              \"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, \" 
              \"%4, %5}, {%6, %7}, {%8, %9};\
\" 
              : \"=r\"(R_S[0][j][0]), \"=r\"(R_S[0][j][1]) 
              : \"r\"(R_Q[0][0]), \"r\"(R_Q[0][1]), \"r\"(R_Q[0][2]), \"r\"(R_Q[0][3]), \"r\"(R_K[j][0]), \"r\"(R_K[j][1]), \"r\"(R_S[0][j][0]), 
                \"r\"(R_S[0][j][1]));
        }
      }
    }
    __syncthreads();

    if ((tile_K_seqlen + 1) < Tc) {
      load_gmem_K_Bc_offset = (tile_K_seqlen + 1) * Bc;
      int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
      int load_gmem_K_d = load_smem_K_d;
      int load_gmem_K_addr =
          (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
      uint32_t load_smem_K_ptr =
          (smem_K_base_ptr + (load_smem_K_Bc * kHeadDim + load_smem_K_d) * sizeof(half));#pragma unroll
      for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
        asm volatile( 
            \"cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\
\" ::\"r\"(load_smem_K_ptr + i * 2), 
            \"l\"(&K[load_gmem_K_addr + i]), \"n\"(16));
      }
      asm volatile(\"cp.async.commit_group;\
\" ::);
    }

    float lane_row_max_new[kWarpTileSeqLenQ][2];
    float lane_row_sum_new[kWarpTileSeqLenQ][2];
    {#pragma unroll
      for (int i = 0; i < kWarpTileSeqLenQ; ++i) {#pragma unroll
        for (int j = 0; j < 2; ++j) {
          lane_row_max_new[i][j] = -INFINITY;
        }
      }
    }
    {#pragma unroll
      for (int i = 0; i < kWarpTileSeqLenQ; ++i) {#pragma unroll
        for (int j = 0; j < 2; ++j) {
          lane_row_sum_new[i][j] = 0.0f;
        }
      }
    }

    static_assert(kWarpTileSeqLenQ == 1);
    {#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) { // 4
        half *t_hptr_S_0_1 = reinterpret_cast<half *>(&(R_S[0][j][0]));
        float tmp_max_0 =
            __half2float(__hmax(t_hptr_S_0_1[0], t_hptr_S_0_1[1])) * scale;
        float tmp_max_1 =
            __half2float(__hmax(t_hptr_S_0_1[2], t_hptr_S_0_1[3])) * scale;
        lane_row_max_new[0][0] = max(lane_row_max_new[0][0], tmp_max_0);
        lane_row_max_new[0][1] = max(lane_row_max_new[0][1], tmp_max_1);
      }

      {
        float _inlined_val_ = lane_row_max_new[0][0];#pragma unroll
        for (int mask = 4 >> 1; mask >= 1; mask >>= 1) {
          _inlined_val_ = max(_inlined_val_, __shfl_xor_sync(0xffffffff, _inlined_val_, mask, 4));
        }
        lane_row_max_new[0][0] = _inlined_val_;
      }
      {
        float _inlined_val_ = lane_row_max_new[0][1];#pragma unroll
        for (int mask = 4 >> 1; mask >= 1; mask >>= 1) {
          _inlined_val_ = max(_inlined_val_, __shfl_xor_sync(0xffffffff, _inlined_val_, mask, 4));
        }
        lane_row_max_new[0][1] = _inlined_val_;
      }
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

      {
        float _inlined_val_ = lane_row_sum_new[0][0];#pragma unroll
        for (int mask = 4 >> 1; mask >= 1; mask >>= 1) {
          _inlined_val_ += __shfl_xor_sync(0xffffffff, _inlined_val_, mask, 4);
        }
        lane_row_sum_new[0][0] = _inlined_val_;
      }
      {
        float _inlined_val_ = lane_row_sum_new[0][1];#pragma unroll
        for (int mask = 4 >> 1; mask >= 1; mask >>= 1) {
          _inlined_val_ += __shfl_xor_sync(0xffffffff, _inlined_val_, mask, 4);
        }
        lane_row_sum_new[0][1] = _inlined_val_;
      }
    }

    if ((tile_K_seqlen + 1) < Tc) {
      asm volatile(\"cp.async.wait_group %0;\
\" ::\"n\"(1));
    } else {
      asm volatile(\"cp.async.wait_group %0;\
\" ::\"n\"(0));
    }
    __syncthreads();

    {#pragma unroll
      for (int i = 0; i < kWarpTileSeqLenP; ++i) {#pragma unroll
        for (int j = 0; j < kWarpTileHeadDimV; ++j) {#pragma unroll
          for (int k = 0; k < 2; ++k) {
            R_O[i][j][k] = 0;
          }
        }
      }
    }#pragma unroll
    for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {#pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        int warp_smem_V_d =
            warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
        int lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16;
        int lane_smem_V_d = warp_smem_V_d;
        uint32_t lane_smem_V_ptr =
            (smem_V_base_ptr + (V_tile_size + lane_smem_V_Bc * kHeadDim + lane_smem_V_d) * sizeof(half));
        asm volatile( 
            \"ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\
\" 
            : \"=r\"(R_V[j][0]), \"=r\"(R_V[j][1]) 
            : \"r\"(lane_smem_V_ptr));
      }

      int w = tile_V_Bc * 2;
      static_assert(kWarpTileSeqLenP == 1);
      {#pragma unroll
        for (int j = 0; j < kWarpTileHeadDimV; ++j) {
          asm volatile( 
              \"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, \" 
              \"%4, %5}, {%6, %7}, {%8, %9};\
\" 
              : \"=r\"(R_O[0][j][0]), \"=r\"(R_O[0][j][1]) 
              : \"r\"(R_S[0][w][0]), \"r\"(R_S[0][w][1]), \"r\"(R_S[0][w + 1][0]), \"r\"(R_S[0][w + 1][1]), \"r\"(R_V[j][0]), \"r\"(R_V[j][1]), \"r\"(R_O[0][j][0]), 
                \"r\"(R_O[0][j][1]));
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
          __expf(block_row_max_old_1 - block_row_max_new_1);#pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        half *t_hptr_O_0_1 = reinterpret_cast<half *>(&(R_O[0][j][0]));
        float *t_fptr_D_0_1 = reinterpret_cast<float *>(&(R_D[0][j][0]));
        t_fptr_D_0_1[0] = __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[0],
                                    __half2float(t_hptr_O_0_1[0]));
        t_fptr_D_0_1[1] = __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[1],
                                    __half2float(t_hptr_O_0_1[1]));
        t_fptr_D_0_1[2] = __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[2],
                                    __half2float(t_hptr_O_0_1[2]));
        t_fptr_D_0_1[3] = __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[3],
                                    __half2float(t_hptr_O_0_1[3]));
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

    if ((tile_K_seqlen + 1) < Tc) {
      asm volatile(\"cp.async.wait_group %0;\
\" ::\"n\"(0));
      __syncthreads();
    }
  }
  __syncthreads();

  static_assert(kWarpTileSeqLenP == 1);
  {
    float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[0][0]);
    float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[0][1]);#pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {
      float *t_fptr_D_0_1 = reinterpret_cast<float *>(&(R_D[0][j][0]));
      half *t_hptr_D_0_1 = reinterpret_cast<half *>(&(R_D[0][j][0]));
      t_hptr_D_0_1[0] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[0]);
      t_hptr_D_0_1[1] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[1]);
      t_hptr_D_0_1[2] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[2]);
      t_hptr_D_0_1[3] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[3]);
    }
  }

  static_assert(kWarpTileSeqLenP == 1);
  {#pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {
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
        (reinterpret_cast<float4 *>(&(O[store_gmem_O_addr_0]))[0]) = ((reinterpret_cast<float4 *>(&(R_Z[0][0]))[0]));
        (reinterpret_cast<float4 *>(&(O[store_gmem_O_addr_1]))[0]) = ((reinterpret_cast<float4 *>(&(R_Z[1][0]))[0]));
      }
    }
  }
}
template <const int kHeadDim>
void launch_flash_attn_mma_stages_split_q_shared_kv(torch::Tensor Q,
                                                    torch::Tensor K,
                                                    torch::Tensor V,
                                                    torch::Tensor O) {
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
  constexpr int kStage = 2;
  constexpr int kMmaTileSeqLenQ = 4;
  constexpr int kMmaTileSeqLenK = 1;
  constexpr int kMmaTileSeqLenP = 4;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  constexpr int kWarpTileSeqLenK = 4;
  constexpr int kWarpTileSeqLenP = 1;
  constexpr int kWarpTileHeadDimV = (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV));
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // 16 * 4 * 1 = 64
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; // 8 * 1 * 4 = 32
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 32 * 4 * 1 = 128  // constexpr int kPadQ = 0; // right but slow  // constexpr int kPadK = 0;  // constexpr int kPadV = 0;
  static_assert(kHeadDim < 256);

  constexpr int Q_tile_size = (Br * kHeadDim);
  constexpr int K_tile_size = (Bc * kHeadDim);
  constexpr int V_tile_size = (Bc * kHeadDim);
  const int smem_max_size =
      (Q_tile_size + kStage * max(K_tile_size, V_tile_size)) * sizeof(half);

  const int QKV_batch = Q.size(0);
  const int QKV_head = Q.size(1);
  const int QKV_seqlen = Q.size(2);
  assert(QKV_seqlen % max(Br, Bc) == 0);
  const float scale = 1.0f / sqrt((float)kHeadDim);

  dim3 grid(((QKV_seqlen % Br != 0) ? (QKV_seqlen / Br + 1) : (QKV_seqlen / Br)), QKV_batch * QKV_head);
  dim3 block(kNumThreads);

  cudaFuncSetAttribute(
      flash_attn_mma_stages_split_q_shared_kv_kernel<
          kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
          kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
          kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
          kStage>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

  flash_attn_mma_stages_split_q_shared_kv_kernel<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
      kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
      kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
      kStage>
      <<<grid, block, smem_max_size>>>(reinterpret_cast<half *>(Q.data_ptr()),
                                       reinterpret_cast<half *>(K.data_ptr()),
                                       reinterpret_cast<half *>(V.data_ptr()),
                                       reinterpret_cast<half *>(O.data_ptr()),
                                       QKV_seqlen, QKV_head, scale);
}

void flash_attn_mma_stages_split_q_shared_kv(torch::Tensor Q, torch::Tensor K,
                                             torch::Tensor V, torch::Tensor O) {
  const int d = Q.size(3);

  switch (d) {
    case 32:
      launch_flash_attn_mma_stages_split_q_shared_kv<32>(Q, K, V, O);
      break;
    case 64:
      launch_flash_attn_mma_stages_split_q_shared_kv<64>(Q, K, V, O);
      break;
    case 96:
      launch_flash_attn_mma_stages_split_q_shared_kv<96>(Q, K, V, O);
      break;
    case 128:
      launch_flash_attn_mma_stages_split_q_shared_kv<128>(Q, K, V, O);
      break;
    default:
      throw std::runtime_error(\"headdim not support!\");
      break;
  }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(\"flash_attn_mma_stages_split_q_shared_kv\",
    &flash_attn_mma_stages_split_q_shared_kv,
    \"flash_attn_mma_stages_split_q_shared_kv\");
}


翻译为triton""")

  model = "gemini-2.5-pro"
  contents = [
    types.Content(
      role="user",
      parts=[
        msg1_text1
      ]
    ),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 0.35,
    top_p = 0.95,
    seed = 0,
    max_output_tokens = 65535,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    thinking_config=types.ThinkingConfig(
      thinking_budget=-1,
    ),
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    print(chunk.text, end="")

generate()