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

// Add DSL template header
#include "dsl_template.cuh"

using namespace nvcuda;

// REMOVED: #define WARP_SIZE 32 (Now provided by dsl_template.cuh)

template <
    const TunableInt kHeadDim, const TunableInt kMmaAtomM, const TunableInt kMmaAtomN,
    const TunableInt kMmaAtomK, const TunableInt kMmaTileSeqLenQ,
    const TunableInt kMmaTileSeqLenK, const TunableInt kMmaTileSeqLenP,
    const TunableInt kMmaTileHeadDimV, const TunableInt kWarpTileSeqLenQ,
    const TunableInt kWarpTileSeqLenK, const TunableInt kWarpTileSeqLenP,
    const TunableInt kWarpTileHeadDimV, const TunableInt kOStorageAccFloat32,
    const TunableInt kStage, const TunableInt kPadQ, const TunableInt kPadK,
    const TunableInt kPadV>
__global__ void __launch_bounds__(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
    flash_attn_mma_stages_split_q_shared_kv_kernel(
        GlobalPtr<fp16> Q, GlobalPtr<fp16> K, GlobalPtr<fp16> V, GlobalPtr<fp16> O,
        ShapeInt QKV_seqlen, ShapeInt QKV_head) {
  // Static asserts remain unchanged
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

  // Use TileInt for computed tile dimensions
  constexpr TileInt Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr TileInt Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  static_assert(Br >= Bc);
  constexpr TileInt kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;

  // Use IndexInt for runtime computed values
  const IndexInt Tc = ceil_div(QKV_seqlen, (ShapeInt)Bc);
  const fp32 scale = 1.0f / sqrt((fp32)kHeadDim);

  // Use IndexInt for thread/block IDs and offsets
  const IndexInt QKV_batch_id = blockIdx.y / QKV_head;
  const IndexInt QKV_head_id = blockIdx.y % QKV_head;
  const IndexInt Q_tile_id = blockIdx.x;
  const IndexInt O_tile_id = Q_tile_id;
  const IndexInt tid = threadIdx.x;
  const IndexInt warp_id = tid / WARP_SIZE;
  const IndexInt lane_id = tid % WARP_SIZE;
  const IndexInt warp_QP = warp_id;
  const IndexInt warp_KV = 0;

  const IndexInt Q_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) + (QKV_head_id * QKV_seqlen * kHeadDim));
  const IndexInt K_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) + (QKV_head_id * QKV_seqlen * kHeadDim));
  const IndexInt V_gmem_offset = Q_gmem_offset;
  const IndexInt O_gmem_offset = Q_gmem_offset;

  IndexInt load_smem_Q_Br = (tid / (kNumThreads / Br));
  IndexInt load_smem_Q_d = (tid % (kNumThreads / Br)) * (kHeadDim / (kNumThreads / Br));
  IndexInt load_smem_K_Bc = (tid / (kNumThreads / Bc));
  IndexInt load_smem_K_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
  IndexInt load_smem_V_Bc = (tid / (kNumThreads / Bc));
  IndexInt load_smem_V_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
  IndexInt load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
  if (load_gmem_Q_Br >= QKV_seqlen) return;
  IndexInt load_gmem_K_Bc_offset = 0;
  IndexInt load_gmem_V_Bc_offset = 0;

  // DSL for dynamic shared memory allocation
  mem_alloc_shared_dynamic(fp16, smem);

  // Use TileInt for shared memory tile sizes
  constexpr TileInt Q_tile_size = Br * (kHeadDim + kPadQ);
  constexpr TileInt K_tile_size = Bc * (kHeadDim + kPadK);
  constexpr TileInt V_tile_size = Bc * (kHeadDim + kPadV);

  // Use semantic SharedPtr for memory access
  SharedPtr<fp16> Q_tile_smem = shared_ptr_cast<fp16>(smem);
  SharedPtr<fp16> K_tile_smem = Q_tile_smem + Q_tile_size;
  SharedPtr<fp16> V_tile_smem = K_tile_smem; // K and V share space

  // Use SmemAddr ONLY for PTX copy instructions
  SmemAddr smem_Q_base_addr = generic_to_shared_addr(Q_tile_smem);
  SmemAddr smem_K_base_addr = generic_to_shared_addr(K_tile_smem);
  SmemAddr smem_V_base_addr = smem_K_base_addr;

  // DSL for register allocation and initialization
  mem_alloc_register(fp32, lane_block_row_max_old, [kWarpTileSeqLenQ][2]);
  mem_fill(lane_block_row_max_old, -INFINITY);

  mem_alloc_register(fp32, lane_block_row_sum_old, [kWarpTileSeqLenQ][2]);
  mem_fill(lane_block_row_sum_old, 0.0f);

  constexpr bool kCanPrefetchQs2r = ((kHeadDim / kMmaAtomK) <= 8) && (kHeadDim < 64);
  constexpr bool kDelayPrefetchQs2r = (true && kCanPrefetchQs2r);
  constexpr bool kCanPrefetchKVg2s = (kStage == 2);
  constexpr int kPrefetchKg2sSmemId = 0;
  constexpr int kPrefetchVg2sSmemId = kCanPrefetchKVg2s ? 1 : 0;
  constexpr int kNumPrefetchQs2r = (kCanPrefetchQs2r) ? (kHeadDim / kMmaAtomK) : 1;

  mem_alloc_register(uint32, R_Q, [kNumPrefetchQs2r][kWarpTileSeqLenQ][4]);
  mem_alloc_register(uint32, R_K, [kWarpTileSeqLenK][2]);
  mem_alloc_register(uint32, R_V, [kWarpTileHeadDimV][2]);
  mem_alloc_register(uint32, R_S, [kWarpTileSeqLenQ][kWarpTileSeqLenK][2]);
  mem_alloc_register(uint32, R_O, [kWarpTileSeqLenP][kWarpTileHeadDimV][2]);
  mem_alloc_register(uint32, R_D, [kWarpTileSeqLenP][kWarpTileHeadDimV][(kOStorageAccFloat32) ? 4 : 2]);
  mem_fill(R_D, 0u);

  // Initial load of Q from global to shared memory
  {
    IndexInt load_gmem_Q_d = load_smem_Q_d;
    IndexInt load_gmem_Q_addr = (Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
    SmemAddr load_smem_Q_addr_base = smem_Q_base_addr + (load_smem_Q_Br * (kHeadDim + kPadQ) + load_gmem_Q_d) * sizeof(fp16);
    serial_range_for(i, 0, (kHeadDim / (kNumThreads / Br)), 8) {
        SmemAddr dst_addr = load_smem_Q_addr_base + i * sizeof(fp16);
        thread_copy_async<fp16, 128, CopyTask::G2S>(&Q[load_gmem_Q_addr + i], dst_addr);
    }
    thread_copy_async_commit_group();
  }

  serial_range_for(tile_K_seqlen, 0, Tc, 1) {
    if constexpr (kCanPrefetchKVg2s) {
      if (tile_K_seqlen == 0) {
        load_gmem_K_Bc_offset = tile_K_seqlen * Bc;
        IndexInt load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
        IndexInt load_gmem_K_d = load_smem_K_d;
        IndexInt load_gmem_K_addr = (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
        SmemAddr load_smem_K_addr_base = smem_K_base_addr + (kPrefetchKg2sSmemId * K_tile_size + load_smem_K_Bc * (kHeadDim + kPadK) + load_gmem_K_d) * sizeof(fp16);
        serial_range_for(i, 0, (kHeadDim / (kNumThreads / Bc)), 8) {
            SmemAddr dst_addr = load_smem_K_addr_base + i * sizeof(fp16);
            thread_copy_async<fp16, 128, CopyTask::G2S>(&K[load_gmem_K_addr + i], dst_addr);
        }
        thread_copy_async_commit_group();
        thread_copy_async_wait_group<0>();
        block_sync();
      }
      {
        load_gmem_V_Bc_offset = tile_K_seqlen * Bc;
        IndexInt load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
        IndexInt load_gmem_V_d = load_smem_V_d;
        IndexInt load_gmem_V_addr = (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
        SmemAddr load_smem_V_addr_base = smem_V_base_addr + (kPrefetchVg2sSmemId * V_tile_size + load_smem_V_Bc * (kHeadDim + kPadV) + load_smem_V_d) * sizeof(fp16);
        serial_range_for(i, 0, (kHeadDim / (kNumThreads / Bc)), 8) {
            SmemAddr dst_addr = load_smem_V_addr_base + i * sizeof(fp16);
            thread_copy_async<fp16, 128, CopyTask::G2S>(&V[load_gmem_V_addr + i], dst_addr);
        }
        thread_copy_async_commit_group();
      }
    } else {
      load_gmem_K_Bc_offset = tile_K_seqlen * Bc;
      IndexInt load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
      IndexInt load_gmem_K_d = load_smem_K_d;
      IndexInt load_gmem_K_addr = (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
      SmemAddr load_smem_K_addr_base = smem_K_base_addr + (kPrefetchKg2sSmemId * K_tile_size + load_smem_K_Bc * (kHeadDim + kPadK) + load_smem_K_d) * sizeof(fp16);
      serial_range_for(i, 0, (kHeadDim / (kNumThreads / Bc)), 8) {
        SmemAddr dst_addr = load_smem_K_addr_base + i * sizeof(fp16);
        thread_copy_async<fp16, 128, CopyTask::G2S>(&K[load_gmem_K_addr + i], dst_addr);
      }
      thread_copy_async_commit_group();
      thread_copy_async_wait_group<0>();
      block_sync();
    }

    if constexpr (kCanPrefetchQs2r && (!kDelayPrefetchQs2r)) {
      if (tile_K_seqlen == 0) {
        if constexpr (!kCanPrefetchKVg2s) { thread_copy_async_wait_group<0>(); } 
        else { thread_copy_async_wait_group<1>(); }
        block_sync();
        serial_range_for(tile_K_d, 0, (kHeadDim / kMmaAtomK), 1) {
          serial_range_for(i, 0, kWarpTileSeqLenQ, 1) {
            IndexInt warp_smem_Q_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
            IndexInt lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
            IndexInt lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
            SharedPtr<fp16> sptr_Q = Q_tile_smem + (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d);
            warp_copy<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(sptr_Q, R_Q[tile_K_d][i][0], R_Q[tile_K_d][i][1], R_Q[tile_K_d][i][2], R_Q[tile_K_d][i][3]);
          }
        }
        block_sync();
      }
    }

    mem_fill(R_S, 0u);
    serial_range_for(tile_K_d, 0, (kHeadDim / kMmaAtomK), 1) {
      if constexpr (!kCanPrefetchQs2r) {
        serial_range_for(i, 0, kWarpTileSeqLenQ, 1) {
            IndexInt warp_smem_Q_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
            IndexInt lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
            IndexInt lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
            SharedPtr<fp16> sptr_Q = Q_tile_smem + (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d);
            warp_copy<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(sptr_Q, R_Q[0][i][0], R_Q[0][i][1], R_Q[0][i][2], R_Q[0][i][3]);
        }
      } else {
        if constexpr (kDelayPrefetchQs2r) {
          if (tile_K_seqlen == 0) {
            if (tile_K_d == 0) {
                if constexpr (!kCanPrefetchKVg2s) { thread_copy_async_wait_group<0>(); } 
                else { thread_copy_async_wait_group<1>(); }
                block_sync();
            }
            serial_range_for(i, 0, kWarpTileSeqLenQ, 1) {
                IndexInt warp_smem_Q_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
                IndexInt lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
                IndexInt lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
                SharedPtr<fp16> sptr_Q = Q_tile_smem + (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d);
                warp_copy<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(sptr_Q, R_Q[tile_K_d][i][0], R_Q[tile_K_d][i][1], R_Q[tile_K_d][i][2], R_Q[tile_K_d][i][3]);
            }
          }
        }
      }

      serial_range_for(j, 0, kWarpTileSeqLenK, 1) {
        IndexInt warp_smem_K_Bc = warp_KV * (kMmaAtomN * kWarpTileSeqLenK) + j * kMmaAtomN;
        IndexInt lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8;
        IndexInt lane_smem_K_d = tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8;
        SharedPtr<fp16> sptr_K = K_tile_smem + (kPrefetchKg2sSmemId * K_tile_size + lane_smem_K_Bc * (kHeadDim + kPadK) + lane_smem_K_d);
        warp_copy<fp16, 64, CopyTask::S2R, Layout::ROW_MAJOR>(sptr_K, R_K[j][0], R_K[j][1]);
      }

      static_assert(kWarpTileSeqLenQ == 1);
      if constexpr (kCanPrefetchQs2r) {
        serial_range_for(j, 0, kWarpTileSeqLenK, 1) {
            mma<MmaShape::M16N8K16, MmaLayout::TN, fp16, fp16>(R_S[0][j][0], R_S[0][j][1], R_Q[tile_K_d][0][0], R_Q[tile_K_d][0][1], R_Q[tile_K_d][0][2], R_Q[tile_K_d][0][3], R_K[j][0], R_K[j][1], R_S[0][j][0], R_S[0][j][1]);
        }
      } else {
        serial_range_for(j, 0, kWarpTileSeqLenK, 1) {
            mma<MmaShape::M16N8K16, MmaLayout::TN, fp16, fp16>(R_S[0][j][0], R_S[0][j][1], R_Q[0][0][0], R_Q[0][0][1], R_Q[0][0][2], R_Q[0][0][3], R_K[j][0], R_K[j][1], R_S[0][j][0], R_S[0][j][1]);
        }
      }
    }
    block_sync();

    if constexpr (!kCanPrefetchKVg2s) {
      load_gmem_V_Bc_offset = tile_K_seqlen * Bc;
      IndexInt load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
      IndexInt load_gmem_V_d = load_smem_V_d;
      IndexInt load_gmem_V_addr = (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
      SmemAddr load_smem_V_addr_base = smem_V_base_addr + (kPrefetchVg2sSmemId * V_tile_size + load_smem_V_Bc * (kHeadDim + kPadV) + load_smem_V_d) * sizeof(fp16);
      serial_range_for(i, 0, (kHeadDim / (kNumThreads / Bc)), 8) {
          SmemAddr dst_addr = load_smem_V_addr_base + i * sizeof(fp16);
          thread_copy_async<fp16, 128, CopyTask::G2S>(&V[load_gmem_V_addr + i], dst_addr);
      }
      thread_copy_async_commit_group();
    }

    if constexpr (kCanPrefetchKVg2s) {
      if ((tile_K_seqlen + 1) < Tc) {
        load_gmem_K_Bc_offset = (tile_K_seqlen + 1) * Bc;
        IndexInt load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
        IndexInt load_gmem_K_d = load_smem_K_d;
        IndexInt load_gmem_K_addr = (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
        SmemAddr load_smem_K_addr_base = smem_K_base_addr + (kPrefetchKg2sSmemId * K_tile_size + load_smem_K_Bc * (kHeadDim + kPadK) + load_gmem_K_d) * sizeof(fp16);
        serial_range_for(i, 0, (kHeadDim / (kNumThreads / Bc)), 8) {
            SmemAddr dst_addr = load_smem_K_addr_base + i * sizeof(fp16);
            thread_copy_async<fp16, 128, CopyTask::G2S>(&K[load_gmem_K_addr + i], dst_addr);
        }
        thread_copy_async_commit_group();
      }
    }

    mem_alloc_register(fp32, lane_row_max_new, [kWarpTileSeqLenQ][2]);
    mem_fill(lane_row_max_new, -INFINITY);
    mem_alloc_register(fp32, lane_row_sum_new, [kWarpTileSeqLenQ][2]);
    mem_fill(lane_row_sum_new, 0.0f);

    static_assert(kWarpTileSeqLenQ == 1);
    {
      serial_range_for(j, 0, kWarpTileSeqLenK, 1) {
        fp16_4* s_vals = reg_cast<fp16_4>(&R_S[0][j][0]);
        fp32 tmp_max_0 = __half2float(__hmax(s_vals->x, s_vals->y)) * scale;
        fp32 tmp_max_1 = __half2float(__hmax(s_vals->z, s_vals->w)) * scale;
        lane_row_max_new[0][0] = max(lane_row_max_new[0][0], tmp_max_0);
        lane_row_max_new[0][1] = max(lane_row_max_new[0][1], tmp_max_1);
      }

      lane_row_max_new[0][0] = warp_shuffle_reduce_quad_max(lane_row_max_new[0][0]);
      lane_row_max_new[0][1] = warp_shuffle_reduce_quad_max(lane_row_max_new[0][1]);
    }

    static_assert(kWarpTileSeqLenQ == 1);
    {
      fp32 block_row_max_new_0 = lane_row_max_new[0][0];
      fp32 block_row_max_new_1 = lane_row_max_new[0][1];
      fp32 block_row_max_old_0 = lane_block_row_max_old[0][0];
      fp32 block_row_max_old_1 = lane_block_row_max_old[0][1];
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

      serial_range_for(j, 0, kWarpTileSeqLenK, 1) {
        fp16_4* s_vals = reg_cast<fp16_4>(&R_S[0][j][0]);
        fp32_4 t_reg_S_0_1;
        t_reg_S_0_1.x = __expf(__fmaf_rn(__half2float(s_vals->x), scale, -block_row_max_new_0));
        t_reg_S_0_1.y = __expf(__fmaf_rn(__half2float(s_vals->y), scale, -block_row_max_new_0));
        t_reg_S_0_1.z = __expf(__fmaf_rn(__half2float(s_vals->z), scale, -block_row_max_new_1));
        t_reg_S_0_1.w = __expf(__fmaf_rn(__half2float(s_vals->w), scale, -block_row_max_new_1));
        lane_row_sum_new[0][0] += (t_reg_S_0_1.x + t_reg_S_0_1.y);
        lane_row_sum_new[0][1] += (t_reg_S_0_1.z + t_reg_S_0_1.w);
        s_vals->x = __float2half_rn(t_reg_S_0_1.x);
        s_vals->y = __float2half_rn(t_reg_S_0_1.y);
        s_vals->z = __float2half_rn(t_reg_S_0_1.z);
        s_vals->w = __float2half_rn(t_reg_S_0_1.w);
      }
      lane_row_sum_new[0][0] = warp_shuffle_reduce_quad_sum(lane_row_sum_new[0][0]);
      lane_row_sum_new[0][1] = warp_shuffle_reduce_quad_sum(lane_row_sum_new[0][1]);
    }

    if constexpr (kCanPrefetchKVg2s) {
        if ((tile_K_seqlen + 1) < Tc) { thread_copy_async_wait_group<1>(); } 
        else { thread_copy_async_wait_group<0>(); }
    } else {
        thread_copy_async_wait_group<0>();
    }
    block_sync();
    
    mem_fill(R_O, 0u);
    serial_range_for(tile_V_Bc, 0, (Bc / kMmaAtomK), 1) {
      serial_range_for(j, 0, kWarpTileHeadDimV, 1) {
        IndexInt warp_smem_V_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
        IndexInt lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16;
        IndexInt lane_smem_V_d = warp_smem_V_d;
        SharedPtr<fp16> sptr_V = V_tile_smem + (kPrefetchVg2sSmemId * V_tile_size + lane_smem_V_Bc * (kHeadDim + kPadV) + lane_smem_V_d);
        warp_copy<fp16, 64, CopyTask::S2R, Layout::COL_MAJOR>(sptr_V, R_V[j][0], R_V[j][1]);
      }

      IndexInt w = tile_V_Bc * 2;
      static_assert(kWarpTileSeqLenP == 1);
      serial_range_for(j, 0, kWarpTileHeadDimV, 1) {
        mma<MmaShape::M16N8K16, MmaLayout::TN, fp16, fp16>(R_O[0][j][0], R_O[0][j][1], R_S[0][w][0], R_S[0][w][1], R_S[0][w + 1][0], R_S[0][w + 1][1], R_V[j][0], R_V[j][1], R_O[0][j][0], R_O[0][j][1]);
      }
    }
    block_sync();

    static_assert(kWarpTileSeqLenP == 1);
    {
      fp32 block_row_max_new_0 = lane_row_max_new[0][0];
      fp32 block_row_max_new_1 = lane_row_max_new[0][1];
      fp32 block_row_sum_new_0 = lane_row_sum_new[0][0];
      fp32 block_row_sum_new_1 = lane_row_sum_new[0][1];
      fp32 block_row_max_old_0 = lane_block_row_max_old[0][0];
      fp32 block_row_max_old_1 = lane_block_row_max_old[0][1];
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);
      block_row_max_old_0 = (tile_K_seqlen > 0 ? block_row_max_old_0 : block_row_max_new_0);
      block_row_max_old_1 = (tile_K_seqlen > 0 ? block_row_max_old_1 : block_row_max_new_1);

      fp32 rescale_o_factor_0 = __expf(block_row_max_old_0 - block_row_max_new_0);
      fp32 rescale_o_factor_1 = __expf(block_row_max_old_1 - block_row_max_new_1);
      serial_range_for(j, 0, kWarpTileHeadDimV, 1) {
        fp16_4* o_vals = reg_cast<fp16_4>(&R_O[0][j][0]);
        if constexpr (kOStorageAccFloat32) {
          fp32_4* d_vals = reg_cast<fp32_4>(&R_D[0][j][0]);
          d_vals->x = __fmaf_rn(rescale_o_factor_0, d_vals->x, __half2float(o_vals->x));
          d_vals->y = __fmaf_rn(rescale_o_factor_0, d_vals->y, __half2float(o_vals->y));
          d_vals->z = __fmaf_rn(rescale_o_factor_1, d_vals->z, __half2float(o_vals->z));
          d_vals->w = __fmaf_rn(rescale_o_factor_1, d_vals->w, __half2float(o_vals->w));
        } else {
          fp16_4* d_vals = reg_cast<fp16_4>(&R_D[0][j][0]);
          d_vals->x = __float2half_rn(__fmaf_rn(rescale_o_factor_0, __half2float(d_vals->x), __half2float(o_vals->x)));
          d_vals->y = __float2half_rn(__fmaf_rn(rescale_o_factor_0, __half2float(d_vals->y), __half2float(o_vals->y)));
          d_vals->z = __float2half_rn(__fmaf_rn(rescale_o_factor_1, __half2float(d_vals->z), __half2float(o_vals->z)));
          d_vals->w = __float2half_rn(__fmaf_rn(rescale_o_factor_1, __half2float(d_vals->w), __half2float(o_vals->w)));
        }
      }

      fp32 block_row_sum_old_0 = lane_block_row_sum_old[0][0];
      fp32 block_row_sum_old_1 = lane_block_row_sum_old[0][1];
      lane_block_row_sum_old[0][0] = __fmaf_rn(rescale_o_factor_0, block_row_sum_old_0, block_row_sum_new_0);
      lane_block_row_sum_old[0][1] = __fmaf_rn(rescale_o_factor_1, block_row_sum_old_1, block_row_sum_new_1);
      lane_block_row_max_old[0][0] = block_row_max_new_0;
      lane_block_row_max_old[0][1] = block_row_max_new_1;
    }

    if constexpr (kCanPrefetchKVg2s) {
      if ((tile_K_seqlen + 1) < Tc) {
        thread_copy_async_wait_group<0>();
        block_sync();
      }
    }
  }
  block_sync();

  static_assert(kWarpTileSeqLenP == 1);
  {
    fp32 rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[0][0]);
    fp32 rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[0][1]);
    serial_range_for(j, 0, kWarpTileHeadDimV, 1) {
      if constexpr (kOStorageAccFloat32) {
        fp32_4* f_d_vals = reg_cast<fp32_4>(&R_D[0][j][0]);
        fp16_4* h_d_vals = reg_cast<fp16_4>(&R_D[0][j][0]);
        h_d_vals->x = __float2half_rn(rescale_factor_0 * f_d_vals->x);
        h_d_vals->y = __float2half_rn(rescale_factor_0 * f_d_vals->y);
        h_d_vals->z = __float2half_rn(rescale_factor_1 * f_d_vals->z);
        h_d_vals->w = __float2half_rn(rescale_factor_1 * f_d_vals->w);
      } else {
        fp16_4* d_vals = reg_cast<fp16_4>(&R_D[0][j][0]);
        d_vals->x = __float2half_rn(rescale_factor_0 * __half2float(d_vals->x));
        d_vals->y = __float2half_rn(rescale_factor_0 * __half2float(d_vals->y));
        d_vals->z = __float2half_rn(rescale_factor_1 * __half2float(d_vals->z));
        d_vals->w = __float2half_rn(rescale_factor_1 * __half2float(d_vals->w));
      }
    }
  }

  static_assert(kWarpTileSeqLenP == 1);
  {
    serial_range_for(j, 0, kWarpTileHeadDimV, 1) {
      if constexpr (kCanPrefetchQs2r && kNumPrefetchQs2r > 1) {
        warp_shuffle_spread_dual_x4(R_D[0][j][0], R_D[0][j][1], R_Q[0][0], R_Q[1][0], lane_id % 4);
        if (lane_id % 4 == 0) {
          IndexInt store_warp_regs_O_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + 0 * kMmaAtomM;
          IndexInt store_lane_gmem_O_Br = O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
          IndexInt store_warp_regs_O_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
          IndexInt store_lane_gmem_O_d = store_warp_regs_O_d;
          IndexInt store_gmem_O_addr_0 = (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim + store_lane_gmem_O_d);
          IndexInt store_gmem_O_addr_1 = (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim + store_lane_gmem_O_d);
          thread_copy<fp32, 128, CopyTask::R2G>(&R_Q[0][0][0], &O[store_gmem_O_addr_0]);
          thread_copy<fp32, 128, CopyTask::R2G>(&R_Q[1][0][0], &O[store_gmem_O_addr_1]);
        }
      } else {
        mem_alloc_register(uint32, R_Z, [2][4]);
        warp_shuffle_spread_dual_x4(R_D[0][j][0], R_D[0][j][1], R_Z[0], R_Z[1], lane_id % 4);
        if (lane_id % 4 == 0) {
          IndexInt store_warp_regs_O_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + 0 * kMmaAtomM;
          IndexInt store_lane_gmem_O_Br = O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
          IndexInt store_warp_regs_O_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
          IndexInt store_lane_gmem_O_d = store_warp_regs_O_d;
          IndexInt store_gmem_O_addr_0 = (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim + store_lane_gmem_O_d);
          IndexInt store_gmem_O_addr_1 = (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim + store_lane_gmem_O_d);
          thread_copy<fp32, 128, CopyTask::R2G>(&R_Z[0][0], &O[store_gmem_O_addr_0]);
          thread_copy<fp32, 128, CopyTask::R2G>(&R_Z[1][0], &O[store_gmem_O_addr_1]);
        }
      }
    }
  }
}

// The launcher function is updated to use the new type system
template <const TunableInt kHeadDim, const TunableInt kStage>
void launch_flash_attn_mma_stages_split_q_shared_kv(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O) {
  constexpr TunableInt kMmaAtomM = 16;
  constexpr TunableInt kMmaAtomN = 8;
  constexpr TunableInt kMmaAtomK = 16;
  constexpr TunableInt kMmaTileSeqLenQ = 4;
  constexpr TunableInt kMmaTileSeqLenK = 1;
  constexpr TunableInt kMmaTileSeqLenP = 4;
  constexpr TunableInt kMmaTileHeadDimV = 1;
  constexpr TunableInt kWarpTileSeqLenQ = 1;
  constexpr TunableInt kWarpTileSeqLenK = (kStage > 1) ? 4 : 8;
  constexpr TunableInt kWarpTileSeqLenP = 1;
  constexpr TunableInt kWarpTileHeadDimV = (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV));
  constexpr TileInt Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr TileInt Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  constexpr TileInt kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  constexpr TunableInt kPadQ = 8;
  constexpr TunableInt kPadK = 8;
  constexpr TunableInt kPadV = 8;
  constexpr TunableInt kOStorageAccFloat32 = (kHeadDim < 256) ? 1 : 0;

  constexpr TileInt Q_tile_size = (Br * (kHeadDim + kPadQ));
  constexpr TileInt K_tile_size = (Bc * (kHeadDim + kPadK));
  constexpr TileInt V_tile_size = (Bc * (kHeadDim + kPadV));
  const IndexInt smem_max_size = (Q_tile_size + kStage * max(K_tile_size, V_tile_size)) * sizeof(fp16);

  const ShapeInt QKV_batch = Q.size(0);
  const ShapeInt QKV_head = Q.size(1);
  const ShapeInt QKV_seqlen = Q.size(2);
  assert(QKV_seqlen % max(Br, Bc) == 0);

  dim3 grid(ceil_div(QKV_seqlen, (ShapeInt)Br), QKV_batch * QKV_head);
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
      <<<grid, block, smem_max_size>>>(
          (GlobalPtr<fp16>)Q.data_ptr(), (GlobalPtr<fp16>)K.data_ptr(),
          (GlobalPtr<fp16>)V.data_ptr(), (GlobalPtr<fp16>)O.data_ptr(),
          QKV_seqlen, QKV_head);
}

// Host-side code remains largely the same, only ensuring type compatibility with launcher
void flash_attn_mma_stages_split_q_shared_kv(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O) {
  if (((Q).options().dtype() != (torch::kHalf))) {
    std::cout << "Tensor Info:" << (Q).options() << std::endl;
    throw std::runtime_error("values must be torch::kHalf");
  }
  if (((K).options().dtype() != (torch::kHalf))) {
    std::cout << "Tensor Info:" << (K).options() << std::endl;
    throw std::runtime_error("values must be torch::kHalf");
  }
  if (((V).options().dtype() != (torch::kHalf))) {
    std::cout << "Tensor Info:" << (V).options() << std::endl;
    throw std::runtime_error("values must be torch::kHalf");
  }
  if (((O).options().dtype() != (torch::kHalf))) {
    std::cout << "Tensor Info:" << (O).options() << std::endl;
    throw std::runtime_error("values must be torch::kHalf");
  }
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
  m.def("flash_attn_mma_stages_split_q_shared_kv",
        &flash_attn_mma_stages_split_q_shared_kv,
        "flash_attn_mma_stages_split_q_shared_kv");
}
