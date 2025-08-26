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

#include "dsl_template.cuh"

template <const TunableInt kHeadDim, const TunableInt kMmaAtomM,
          const TunableInt kMmaAtomN, const TunableInt kMmaAtomK,
          const TunableInt kMmaTileSeqLenQ, const TunableInt kMmaTileSeqLenK,
          const TunableInt kMmaTileSeqLenP, const TunableInt kMmaTileHeadDimV,
          const TunableInt kWarpTileSeqLenQ, const TunableInt kWarpTileSeqLenK,
          const TunableInt kWarpTileSeqLenP, const TunableInt kWarpTileHeadDimV,
          const TunableInt kOStorageAccFloat32, const TunableInt kStage,
          const TunableInt kPadQ, const TunableInt kPadK,
          const TunableInt kPadV>
__global__ void __launch_bounds__(WARP_SIZE *kMmaTileSeqLenQ *kMmaTileSeqLenK)
    flash_attn_mma_stages_split_q_shared_kv_kernel(
        GlobalPtr<fp16> Q, GlobalPtr<fp16> K, GlobalPtr<fp16> V,
        GlobalPtr<fp16> O, ShapeInt QKV_seqlen, ShapeInt QKV_head) {
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

  constexpr TileInt Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr TileInt Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  static_assert(Br >= Bc);
  constexpr TileInt kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;

  const IndexInt Tc = ceil_div(QKV_seqlen, Bc);
  const fp32 scale = 1.0f / sqrt((fp32)kHeadDim);

  const IndexInt QKV_batch_id = blockIdx.y / QKV_head;
  const IndexInt QKV_head_id = blockIdx.y % QKV_head;
  const IndexInt Q_tile_id = blockIdx.x;
  const IndexInt O_tile_id = Q_tile_id;
  const IndexInt tid = threadIdx.x;
  const IndexInt warp_id = tid / WARP_SIZE;
  const IndexInt lane_id = tid % WARP_SIZE;
  const IndexInt warp_QP = warp_id;
  const IndexInt warp_KV = 0;
  const IndexInt Q_gmem_offset =
      ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
       (QKV_head_id * QKV_seqlen * kHeadDim));
  const IndexInt K_gmem_offset =
      ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
       (QKV_head_id * QKV_seqlen * kHeadDim));
  const IndexInt V_gmem_offset = Q_gmem_offset;
  const IndexInt O_gmem_offset = Q_gmem_offset;

  IndexInt load_smem_Q_Br = (tid / (kNumThreads / Br));
  IndexInt load_smem_Q_d =
      (tid % (kNumThreads / Br)) * (kHeadDim / (kNumThreads / Br));
  IndexInt load_smem_K_Bc = (tid / (kNumThreads / Bc));
  IndexInt load_smem_K_d =
      (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
  IndexInt load_smem_V_Bc = (tid / (kNumThreads / Bc));
  IndexInt load_smem_V_d =
      (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
  IndexInt load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
  if (load_gmem_Q_Br >= QKV_seqlen) return;
  IndexInt load_gmem_K_Bc_offset = 0;
  IndexInt load_gmem_V_Bc_offset = 0;

  mem_alloc_shared_dynamic(fp16, smem);

  constexpr TileInt Q_tile_size = Br * (kHeadDim + kPadQ);
  constexpr TileInt K_tile_size = Bc * (kHeadDim + kPadK);
  constexpr TileInt V_tile_size = Bc * (kHeadDim + kPadV);

  SharedPtr<fp16> Q_tile_smem = shared_ptr_cast<fp16>(smem);
  SharedPtr<fp16> K_tile_smem = Q_tile_smem + Q_tile_size;
  SharedPtr<fp16> V_tile_smem = K_tile_smem;

  SmemAddr smem_Q_base_ptr = generic_to_shared_addr(Q_tile_smem);
  SmemAddr smem_K_base_ptr = generic_to_shared_addr(K_tile_smem);
  SmemAddr smem_V_base_ptr = generic_to_shared_addr(V_tile_smem);

  mem_alloc_register(fp32, lane_block_row_max_old, [kWarpTileSeqLenQ][2]);
  mem_alloc_register(fp32, lane_block_row_sum_old, [kWarpTileSeqLenQ][2]);

  mem_fill(lane_block_row_max_old, -INFINITY);
  mem_fill(lane_block_row_sum_old, 0.0f);
  constexpr bool kCanPrefetchQs2r =
      ((kHeadDim / kMmaAtomK) <= 8) && (kHeadDim < 64);
  constexpr bool kDelayPrefetchQs2r = (true && kCanPrefetchQs2r);
  constexpr bool kCanPrefetchKVg2s = (kStage == 2);

  constexpr TileInt kPrefetchKg2sSmemId = 0;
  constexpr TileInt kPrefetchVg2sSmemId = kCanPrefetchKVg2s ? 1 : 0;
  constexpr TileInt kNumPrefetchQs2r =
      (kCanPrefetchQs2r) ? (kHeadDim / kMmaAtomK) : 1;

  mem_alloc_register(uint32, R_Q, [kNumPrefetchQs2r][kWarpTileSeqLenQ][4]);
  mem_alloc_register(uint32, R_K, [kWarpTileSeqLenK][2]);
  mem_alloc_register(uint32, R_V, [kWarpTileHeadDimV][2]);
  mem_alloc_register(uint32, R_S, [kWarpTileSeqLenQ][kWarpTileSeqLenK][2]);
  mem_alloc_register(uint32, R_O, [kWarpTileSeqLenP][kWarpTileHeadDimV][2]);
  mem_alloc_register(
      uint32, R_D,
      [kWarpTileSeqLenP][kWarpTileHeadDimV][(kOStorageAccFloat32) ? 4 : 2]);

  mem_fill(R_D, 0u);
  {
    IndexInt load_gmem_Q_d = load_smem_Q_d;
    IndexInt load_gmem_Q_addr =
        (Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);

    SmemAddr load_smem_Q_ptr =
        (smem_Q_base_ptr +
         (load_smem_Q_Br * (kHeadDim + kPadQ) + load_smem_Q_d) * sizeof(fp16));

    serial_range_for(i, 0, (kHeadDim / (kNumThreads / Br)), 8) {
      thread_copy_async<fp16, 128, CopyTask::G2S>(
          &Q[load_gmem_Q_addr + i], load_smem_Q_ptr + i * sizeof(fp16));
    }
    thread_copy_async_commit_group();
  }
  serial_range_for(tile_K_seqlen, 0, Tc, 1) {
    if constexpr (kCanPrefetchKVg2s) {
      if (tile_K_seqlen == 0) {
        load_gmem_K_Bc_offset = tile_K_seqlen * Bc;
        IndexInt load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
        IndexInt load_gmem_K_d = load_smem_K_d;
        IndexInt load_gmem_K_addr =
            (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);

        SmemAddr load_smem_K_ptr =
            (smem_K_base_ptr +
             (kPrefetchKg2sSmemId * K_tile_size +
              load_smem_K_Bc * (kHeadDim + kPadK) + load_smem_K_d) *
                 sizeof(fp16));

        serial_range_for(i, 0, (kHeadDim / (kNumThreads / Bc)), 8) {
          thread_copy_async<fp16, 128, CopyTask::G2S>(
              &K[load_gmem_K_addr + i], load_smem_K_ptr + i * sizeof(fp16));
        }
        thread_copy_async_commit_group();

        thread_copy_async_wait_group<0>();
        block_sync();
      }
      {
        load_gmem_V_Bc_offset = tile_K_seqlen * Bc;
        IndexInt load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
        IndexInt load_gmem_V_d = load_smem_V_d;
        IndexInt load_gmem_V_addr =
            (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);

        SmemAddr load_smem_V_ptr =
            (smem_V_base_ptr +
             (kPrefetchVg2sSmemId * V_tile_size +
              load_smem_V_Bc * (kHeadDim + kPadV) + load_smem_V_d) *
                 sizeof(fp16));

        serial_range_for(i, 0, (kHeadDim / (kNumThreads / Bc)), 8) {
          thread_copy_async<fp16, 128, CopyTask::G2S>(
              &V[load_gmem_V_addr + i], load_smem_V_ptr + i * sizeof(fp16));
        }
        thread_copy_async_commit_group();
      }
    } else {
      load_gmem_K_Bc_offset = tile_K_seqlen * Bc;
      IndexInt load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
      IndexInt load_gmem_K_d = load_smem_K_d;
      IndexInt load_gmem_K_addr =
          (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);

      SmemAddr load_smem_K_ptr =
          (smem_K_base_ptr +
           (kPrefetchKg2sSmemId * K_tile_size +
            load_smem_K_Bc * (kHeadDim + kPadK) + load_smem_K_d) *
               sizeof(fp16));

      serial_range_for(i, 0, (kHeadDim / (kNumThreads / Bc)), 8) {
        thread_copy_async<fp16, 128, CopyTask::G2S>(
            &K[load_gmem_K_addr + i], load_smem_K_ptr + i * sizeof(fp16));
      }
      thread_copy_async_commit_group();
      thread_copy_async_wait_group<0>();
      block_sync();
    }
    if constexpr (kCanPrefetchQs2r && (!kDelayPrefetchQs2r)) {
      if (tile_K_seqlen == 0) {
        if constexpr (!kCanPrefetchKVg2s) {
          thread_copy_async_wait_group<0>();
        } else {
          thread_copy_async_wait_group<1>();
        }
        block_sync();

        serial_range_for(tile_K_d, 0, (kHeadDim / kMmaAtomK), 1) {
          serial_range_for(i, 0, kWarpTileSeqLenQ, 1) {
            IndexInt warp_smem_Q_Br =
                warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
            IndexInt lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
            IndexInt lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;

            SharedPtr<fp16> lane_smem_Q_sptr =
                Q_tile_smem +
                (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d);

            warp_copy<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(
                lane_smem_Q_sptr, R_Q[tile_K_d][i][0], R_Q[tile_K_d][i][1],
                R_Q[tile_K_d][i][2], R_Q[tile_K_d][i][3]);
          }
        }
        block_sync();
      }
    }
    { mem_fill(R_S, 0u); }
    serial_range_for(tile_K_d, 0, (kHeadDim / kMmaAtomK), 1) {
      if constexpr (!kCanPrefetchQs2r) {
        serial_range_for(i, 0, kWarpTileSeqLenQ, 1) {
          IndexInt warp_smem_Q_Br =
              warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
          IndexInt lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
          IndexInt lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;

          SharedPtr<fp16> lane_smem_Q_sptr =
              Q_tile_smem +
              (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d);

          warp_copy<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(
              lane_smem_Q_sptr, R_Q[0][i][0], R_Q[0][i][1], R_Q[0][i][2],
              R_Q[0][i][3]);
        }
      } else {
        if constexpr (kDelayPrefetchQs2r) {
          if (tile_K_seqlen == 0) {
            if (tile_K_d == 0) {
              if constexpr (!kCanPrefetchKVg2s) {
                thread_copy_async_wait_group<0>();
              } else {
                thread_copy_async_wait_group<1>();
              }
              block_sync();
            }
            serial_range_for(i, 0, kWarpTileSeqLenQ, 1) {
              IndexInt warp_smem_Q_Br =
                  warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
              IndexInt lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
              IndexInt lane_smem_Q_d =
                  tile_K_d * kMmaAtomK + (lane_id / 16) * 8;

              SharedPtr<fp16> lane_smem_Q_sptr =
                  Q_tile_smem +
                  (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d);

              warp_copy<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(
                  lane_smem_Q_sptr, R_Q[tile_K_d][i][0], R_Q[tile_K_d][i][1],
                  R_Q[tile_K_d][i][2], R_Q[tile_K_d][i][3]);
            }
          }
        }
      }
      serial_range_for(j, 0, kWarpTileSeqLenK, 1) {
        IndexInt warp_smem_K_Bc =
            warp_KV * (kMmaAtomN * kWarpTileSeqLenK) + j * kMmaAtomN;
        IndexInt lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8;
        IndexInt lane_smem_K_d = tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8;

        SharedPtr<fp16> lane_smem_K_sptr =
            K_tile_smem + (kPrefetchKg2sSmemId * K_tile_size +
                           lane_smem_K_Bc * (kHeadDim + kPadK) + lane_smem_K_d);

        warp_copy<fp16, 64, CopyTask::S2R, Layout::ROW_MAJOR>(
            lane_smem_K_sptr, R_K[j][0], R_K[j][1]);
      }
      if constexpr (kCanPrefetchQs2r) {
        static_assert(kWarpTileSeqLenQ == 1);
        {
          serial_range_for(j, 0, kWarpTileSeqLenK, 1) {
            mma<MmaShape::M16N8K16, MmaLayout::TN, fp16, fp16>(
                R_S[0][j][0], R_S[0][j][1], R_Q[tile_K_d][0][0],
                R_Q[tile_K_d][0][1], R_Q[tile_K_d][0][2], R_Q[tile_K_d][0][3],
                R_K[j][0], R_K[j][1], R_S[0][j][0], R_S[0][j][1]);
          }
        }
      } else {
        static_assert(kWarpTileSeqLenQ == 1);
        {
          serial_range_for(j, 0, kWarpTileSeqLenK, 1) {
            mma<MmaShape::M16N8K16, MmaLayout::TN, fp16, fp16>(
                R_S[0][j][0], R_S[0][j][1], R_Q[0][0][0], R_Q[0][0][1],
                R_Q[0][0][2], R_Q[0][0][3], R_K[j][0], R_K[j][1], R_S[0][j][0],
                R_S[0][j][1]);
          }
        }
      }
    }
    block_sync();
    if constexpr (!kCanPrefetchKVg2s) {
      load_gmem_V_Bc_offset = tile_K_seqlen * Bc;
      IndexInt load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
      IndexInt load_gmem_V_d = load_smem_V_d;
      IndexInt load_gmem_V_addr =
          (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);

      SmemAddr load_smem_V_ptr =
          (smem_V_base_ptr +
           (kPrefetchVg2sSmemId * V_tile_size +
            load_smem_V_Bc * (kHeadDim + kPadV) + load_smem_V_d) *
               sizeof(fp16));

      serial_range_for(i, 0, (kHeadDim / (kNumThreads / Bc)), 8) {
        thread_copy_async<fp16, 128, CopyTask::G2S>(
            &V[load_gmem_V_addr + i], load_smem_V_ptr + i * sizeof(fp16));
      }
      thread_copy_async_commit_group();
    }
    if constexpr (kCanPrefetchKVg2s) {
      if ((tile_K_seqlen + 1) < Tc) {
        load_gmem_K_Bc_offset = (tile_K_seqlen + 1) * Bc;
        IndexInt load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
        IndexInt load_gmem_K_d = load_smem_K_d;
        IndexInt load_gmem_K_addr =
            (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);

        SmemAddr load_smem_K_ptr =
            (smem_K_base_ptr +
             (kPrefetchKg2sSmemId * K_tile_size +
              load_smem_K_Bc * (kHeadDim + kPadK) + load_smem_K_d) *
                 sizeof(fp16));

        serial_range_for(i, 0, (kHeadDim / (kNumThreads / Bc)), 8) {
          thread_copy_async<fp16, 128, CopyTask::G2S>(
              &K[load_gmem_K_addr + i], load_smem_K_ptr + i * sizeof(fp16));
        }
        thread_copy_async_commit_group();
      }
    }
    mem_alloc_register(fp32, lane_row_max_new, [kWarpTileSeqLenQ][2]);
    mem_alloc_register(fp32, lane_row_sum_new, [kWarpTileSeqLenQ][2]);

    mem_fill(lane_row_max_new, -INFINITY);
    mem_fill(lane_row_sum_new, 0.0f);
    static_assert(kWarpTileSeqLenQ == 1);
    {
      serial_range_for(j, 0, kWarpTileSeqLenK, 1) {
        RegisterPtr<fp16> t_hptr_S = register_ptr_cast<fp16>(R_S[0][j]);

        fp32 tmp_max_0 = __half2float(__hmax(t_hptr_S[0], t_hptr_S[1])) * scale;
        fp32 tmp_max_1 = __half2float(__hmax(t_hptr_S[2], t_hptr_S[3])) * scale;

        lane_row_max_new[0][0] = fmaxf(lane_row_max_new[0][0], tmp_max_0);
        lane_row_max_new[0][1] = fmaxf(lane_row_max_new[0][1], tmp_max_1);
      }

      lane_row_max_new[0][0] =
          warp_shuffle_max_width<fp32>(lane_row_max_new[0][0], 4);
      lane_row_max_new[0][1] =
          warp_shuffle_max_width<fp32>(lane_row_max_new[0][1], 4);
    }
    static_assert(kWarpTileSeqLenQ == 1);
    {
      fp32 block_row_max_new_0 = lane_row_max_new[0][0];
      fp32 block_row_max_new_1 = lane_row_max_new[0][1];

      fp32 block_row_max_old_0 = lane_block_row_max_old[0][0];
      fp32 block_row_max_old_1 = lane_block_row_max_old[0][1];

      block_row_max_new_0 = fmaxf(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = fmaxf(block_row_max_old_1, block_row_max_new_1);

      serial_range_for(j, 0, kWarpTileSeqLenK, 1) {
        RegisterPtr<fp16> t_hptr_S = register_ptr_cast<fp16>(R_S[0][j]);

        fp32_4 t_reg_S;

        t_reg_S.x = __expf(
            __fmaf_rn(__half2float(t_hptr_S[0]), scale, -block_row_max_new_0));
        t_reg_S.y = __expf(
            __fmaf_rn(__half2float(t_hptr_S[1]), scale, -block_row_max_new_0));
        t_reg_S.z = __expf(
            __fmaf_rn(__half2float(t_hptr_S[2]), scale, -block_row_max_new_1));
        t_reg_S.w = __expf(
            __fmaf_rn(__half2float(t_hptr_S[3]), scale, -block_row_max_new_1));

        lane_row_sum_new[0][0] += (t_reg_S.x + t_reg_S.y);
        lane_row_sum_new[0][1] += (t_reg_S.z + t_reg_S.w);

        t_hptr_S[0] = __float2half_rn(t_reg_S.x);
        t_hptr_S[1] = __float2half_rn(t_reg_S.y);
        t_hptr_S[2] = __float2half_rn(t_reg_S.z);
        t_hptr_S[3] = __float2half_rn(t_reg_S.w);
      }

      lane_row_sum_new[0][0] =
          warp_shuffle_sum_width<fp32>(lane_row_sum_new[0][0], 4);
      lane_row_sum_new[0][1] =
          warp_shuffle_sum_width<fp32>(lane_row_sum_new[0][1], 4);
    }
    if constexpr (kCanPrefetchKVg2s) {
      if ((tile_K_seqlen + 1) < Tc) {
        thread_copy_async_wait_group<1>();
      } else {
        thread_copy_async_wait_group<0>();
      }
    } else {
      thread_copy_async_wait_group<0>();
    }
    block_sync();
    { mem_fill(R_O, 0u); }
    serial_range_for(tile_V_Bc, 0, (Bc / kMmaAtomK), 1) {
      serial_range_for(j, 0, kWarpTileHeadDimV, 1) {
        IndexInt warp_smem_V_d =
            warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
        IndexInt lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16;
        IndexInt lane_smem_V_d = warp_smem_V_d;

        SharedPtr<fp16> lane_smem_V_sptr =
            V_tile_smem + (kPrefetchVg2sSmemId * V_tile_size +
                           lane_smem_V_Bc * (kHeadDim + kPadV) + lane_smem_V_d);

        warp_copy<fp16, 64, CopyTask::S2R, Layout::COL_MAJOR>(
            lane_smem_V_sptr, R_V[j][0], R_V[j][1]);
      }

      IndexInt w = tile_V_Bc * 2;
      static_assert(kWarpTileSeqLenP == 1);
      {
        serial_range_for(j, 0, kWarpTileHeadDimV, 1) {
          mma<MmaShape::M16N8K16, MmaLayout::TN, fp16, fp16>(
              R_O[0][j][0], R_O[0][j][1], R_S[0][w][0], R_S[0][w][1],
              R_S[0][w + 1][0], R_S[0][w + 1][1], R_V[j][0], R_V[j][1],
              R_O[0][j][0], R_O[0][j][1]);
        }
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

      block_row_max_new_0 = fmaxf(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = fmaxf(block_row_max_old_1, block_row_max_new_1);

      block_row_max_old_0 =
          (tile_K_seqlen > 0 ? block_row_max_old_0 : block_row_max_new_0);
      block_row_max_old_1 =
          (tile_K_seqlen > 0 ? block_row_max_old_1 : block_row_max_new_1);

      fp32 rescale_o_factor_0 =
          __expf(block_row_max_old_0 - block_row_max_new_0);
      fp32 rescale_o_factor_1 =
          __expf(block_row_max_old_1 - block_row_max_new_1);

      serial_range_for(j, 0, kWarpTileHeadDimV, 1) {
        RegisterPtr<fp16> t_hptr_O = register_ptr_cast<fp16>(R_O[0][j]);

        if constexpr (kOStorageAccFloat32) {
          RegisterPtr<fp32> t_fptr_D = register_ptr_cast<fp32>(R_D[0][j]);
          t_fptr_D[0] = __fmaf_rn(rescale_o_factor_0, t_fptr_D[0],
                                  __half2float(t_hptr_O[0]));
          t_fptr_D[1] = __fmaf_rn(rescale_o_factor_0, t_fptr_D[1],
                                  __half2float(t_hptr_O[1]));
          t_fptr_D[2] = __fmaf_rn(rescale_o_factor_1, t_fptr_D[2],
                                  __half2float(t_hptr_O[2]));
          t_fptr_D[3] = __fmaf_rn(rescale_o_factor_1, t_fptr_D[3],
                                  __half2float(t_hptr_O[3]));
        } else {
          RegisterPtr<fp16> t_hptr_D = register_ptr_cast<fp16>(R_D[0][j]);
          t_hptr_D[0] = __float2half_rn(__fmaf_rn(rescale_o_factor_0,
                                                  __half2float(t_hptr_D[0]),
                                                  __half2float(t_hptr_O[0])));
          t_hptr_D[1] = __float2half_rn(__fmaf_rn(rescale_o_factor_0,
                                                  __half2float(t_hptr_D[1]),
                                                  __half2float(t_hptr_O[1])));
          t_hptr_D[2] = __float2half_rn(__fmaf_rn(rescale_o_factor_1,
                                                  __half2float(t_hptr_D[2]),
                                                  __half2float(t_hptr_O[2])));
          t_hptr_D[3] = __float2half_rn(__fmaf_rn(rescale_o_factor_1,
                                                  __half2float(t_hptr_D[3]),
                                                  __half2float(t_hptr_O[3])));
        }
      }

      fp32 block_row_sum_old_0 = lane_block_row_sum_old[0][0];
      fp32 block_row_sum_old_1 = lane_block_row_sum_old[0][1];
      lane_block_row_sum_old[0][0] = (__fmaf_rn(
          rescale_o_factor_0, block_row_sum_old_0, block_row_sum_new_0));
      lane_block_row_sum_old[0][1] = (__fmaf_rn(
          rescale_o_factor_1, block_row_sum_old_1, block_row_sum_new_1));
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
        RegisterPtr<fp32> t_fptr_D = register_ptr_cast<fp32>(R_D[0][j]);
        RegisterPtr<fp16> t_hptr_D = register_ptr_cast<fp16>(R_D[0][j]);

        t_hptr_D[0] = __float2half_rn(rescale_factor_0 * t_fptr_D[0]);
        t_hptr_D[1] = __float2half_rn(rescale_factor_0 * t_fptr_D[1]);
        t_hptr_D[2] = __float2half_rn(rescale_factor_1 * t_fptr_D[2]);
        t_hptr_D[3] = __float2half_rn(rescale_factor_1 * t_fptr_D[3]);
      } else {
        RegisterPtr<fp16> t_hptr_D = register_ptr_cast<fp16>(R_D[0][j]);

        t_hptr_D[0] =
            __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D[0]));
        t_hptr_D[1] =
            __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D[1]));
        t_hptr_D[2] =
            __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D[2]));
        t_hptr_D[3] =
            __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D[3]));
      }
    }
  }
  static_assert(kWarpTileSeqLenP == 1);
  {
    serial_range_for(j, 0, kWarpTileHeadDimV, 1) {
      if constexpr (kCanPrefetchQs2r && kNumPrefetchQs2r > 1) {
        warp_shuffle_spread_dual_x4(R_D[0][j][0], R_D[0][j][1], R_Q[0][0],
                                    R_Q[1][0], lane_id, 4);

        if (lane_id % 4 == 0) {
          IndexInt store_warp_regs_O_Br =
              warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + 0 * kMmaAtomM;
          IndexInt store_lane_gmem_O_Br =
              O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
          IndexInt store_warp_regs_O_d =
              warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
          IndexInt store_lane_gmem_O_d = store_warp_regs_O_d;
          IndexInt store_gmem_O_addr_0 =
              (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim +
               store_lane_gmem_O_d);
          IndexInt store_gmem_O_addr_1 =
              (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim +
               store_lane_gmem_O_d);

          thread_copy<fp16, 128, CopyTask::R2G>(&R_Q[0][0][0],
                                                &O[store_gmem_O_addr_0]);
          thread_copy<fp16, 128, CopyTask::R2G>(&R_Q[1][0][0],
                                                &O[store_gmem_O_addr_1]);
        }
      } else {
        mem_alloc_register(uint32, R_Z, [2][4]);

        warp_shuffle_spread_dual_x4(R_D[0][j][0], R_D[0][j][1], R_Z[0], R_Z[1],
                                    lane_id, 4);

        if (lane_id % 4 == 0) {
          IndexInt store_warp_regs_O_Br =
              warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + 0 * kMmaAtomM;
          IndexInt store_lane_gmem_O_Br =
              O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
          IndexInt store_warp_regs_O_d =
              warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
          IndexInt store_lane_gmem_O_d = store_warp_regs_O_d;
          IndexInt store_gmem_O_addr_0 =
              (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim +
               store_lane_gmem_O_d);
          IndexInt store_gmem_O_addr_1 =
              (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim +
               store_lane_gmem_O_d);

          thread_copy<fp16, 128, CopyTask::R2G>(&R_Z[0][0],
                                                &O[store_gmem_O_addr_0]);
          thread_copy<fp16, 128, CopyTask::R2G>(&R_Z[1][0],
                                                &O[store_gmem_O_addr_1]);
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

  dim3 grid(
      ((QKV_seqlen % Br != 0) ? (QKV_seqlen / Br + 1) : (QKV_seqlen / Br)),
      QKV_batch * QKV_head);
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
  if (((Q).options().dtype() != (torch::kHalf))) {
    std::cout << "Tensor Info:" << (Q).options() << std::endl;
    throw std::runtime_error(
        "values must be "
        "torch::kHalf");
  }
  if (((K).options().dtype() != (torch::kHalf))) {
    std::cout << "Tensor Info:" << (K).options() << std::endl;
    throw std::runtime_error(
        "values must be "
        "torch::kHalf");
  }
  if (((V).options().dtype() != (torch::kHalf))) {
    std::cout << "Tensor Info:" << (V).options() << std::endl;
    throw std::runtime_error(
        "values must be "
        "torch::kHalf");
  }
  if (((O).options().dtype() != (torch::kHalf))) {
    std::cout << "Tensor Info:" << (O).options() << std::endl;
    throw std::runtime_error(
        "values must be "
        "torch::kHalf");
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