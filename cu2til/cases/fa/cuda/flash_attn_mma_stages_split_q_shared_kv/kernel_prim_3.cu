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

// Include the DSL template header
#include "dsl_template.cuh"

using namespace nvcuda;

template <const TunableInt kHeadDim, const TunableInt kMmaAtomM = 16, const TunableInt kMmaAtomN = 8,
          const TunableInt kMmaAtomK = 16, const TunableInt kMmaTileSeqLenQ = 4,
          const TunableInt kMmaTileSeqLenK = 1, const TunableInt kMmaTileSeqLenP = 4,
          const TunableInt kMmaTileHeadDimV = 1, const TunableInt kWarpTileSeqLenQ = 1,
          const TunableInt kWarpTileSeqLenK = 4, const TunableInt kWarpTileSeqLenP = 1,
          const TunableInt kWarpTileHeadDimV = (kHeadDim / 8),
          const TunableInt kStage = 2>
__global__ void __launch_bounds__(WARP_SIZE *kMmaTileSeqLenQ *kMmaTileSeqLenK)
    flash_attn_mma_stages_split_q_shared_kv_kernel(GlobalPtr<fp16> Q, GlobalPtr<fp16> K, GlobalPtr<fp16> V,
                                                   GlobalPtr<fp16> O, ShapeInt QKV_seqlen,
                                                   ShapeInt QKV_head, fp32 scale) {
  // Static assertions for kernel constraints
  static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMmaAtomK == 16);
  static_assert(kMmaTileSeqLenQ <= 8 && kMmaTileSeqLenK == 1);
  static_assert(kMmaTileSeqLenP <= 8 && kMmaTileHeadDimV == 1);
  static_assert(kWarpTileSeqLenQ == 1 && kWarpTileSeqLenK <= 16);
  static_assert(kWarpTileSeqLenP == 1 && kWarpTileHeadDimV == (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)));
  static_assert(kStage == 2);

  // Tile-level dimension calculations
  constexpr TileInt Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr TileInt Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  static_assert(Br >= Bc);
  constexpr TileInt kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  assert(QKV_seqlen % Bc == 0);
  const IndexInt Tc = QKV_seqlen / Bc;

  // Thread and block indexing
  const IdInt block_id_x = blockIdx.x;
  const IdInt block_id_y = blockIdx.y;
  const IndexInt QKV_batch_id = block_id_y / QKV_head;
  const IndexInt QKV_head_id = block_id_y % QKV_head;
  const IndexInt Q_tile_id = block_id_x;
  const IndexInt O_tile_id = Q_tile_id;
  const IdInt tid = threadIdx.x; // 0...127
  const IdInt warp_id = tid / WARP_SIZE; // 0...4
  const IdInt lane_id = tid % WARP_SIZE; // 0...31
  const NumInt warp_num = 4;
  const IndexInt warp_QP = warp_id;
  const IndexInt warp_KV = 0;

  // Global memory offset calculations
  const IndexInt Q_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) + (QKV_head_id * QKV_seqlen * kHeadDim));
    // make_coord_style_offset(coord=(batch_id, head_id), stride=(QKV_head, 1), step_tile=(QKV_seqlen, kHeadDim))
  const IndexInt K_gmem_offset = Q_gmem_offset;
  const IndexInt V_gmem_offset = Q_gmem_offset;
  const IndexInt O_gmem_offset = Q_gmem_offset;

  // Thread-to-data mapping for shared memory loading
  // (kNumThreads / Br) = (128 / 64) = 2: (Br, kHeadDim)的smem块每行有 2 个线程处理
  IndexInt load_smem_Q_Br = (tid / (kNumThreads / Br));// 标记当前tid在哪个Br块中, tid:(0,...,kNumThreads-1)
  IndexInt load_smem_Q_d = (tid % (kNumThreads / Br)) * (kHeadDim / (kNumThreads / Br)); // (tid % 2) * (d / 2)
  IndexInt load_smem_K_Bc = (tid / (kNumThreads / Bc)); // [0 : Bc]
  // [0 : kNumThreads] / (kNumThreads / Bc) => [0 : Bc] : (kNumThreads / Bc)
  // [0 : kNumThreads / Bc] 映射到0, [kNumThreads / Bc : 2 * kNumThreads / Bc] 映射到1, ...
  // [0 : 128] / (128 / 32) => (32, 4):(4, 1) => [0 : 32][0 : 4] 感觉光从这一行推不出来的
  // coord of smem of q tid -> (load_smem_Q_Br, load_smem_Q_d) => (tid / 2, (tid % 2) * col/2 )
  // 可以让llm尝试推断模式, 失败则保留原代码
  // load_smem_Q_Br: block_tile(Br, kHeadDim)的行id(每个线程都对应一个行id)
  // load_smem_Q_d: block_tile(Br, kHeadDim)的列id(每个线程都对应一个列id)
  // 这是一个(Br, 128/Br):(128/Br, 1)=>(Br, 2):(2, 1) -> (1, d/2) of Tile(Br, kHeadDim)的映射
  IndexInt load_smem_K_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
  // coord(tid / (kNumThreads / Bc), tid % (kNumThreads / Bc)) 
  // => [0 : kNumThreads] / (kNumThreads / Bc), [0 : kNumThreads] % (kNumThreads / Bc) 
  // => [0 : Bc][0 : (kNumThreads / Bc)]:{(kNumThreads / Bc), 1}
  // => [0 : Bc][0 : kHeadDim : (kHeadDim / (kNumThreads / Bc))]:((kNumThreads / Bc), 1)
  // [addr_coord_dim1][addr_coord_dim2]:(coord_stride_dim1, coord_stride_dim2)
  // coord(2addr)_space:tid2coord_stride
  IndexInt load_smem_V_Bc = (tid / (kNumThreads / Bc));
  IndexInt load_smem_V_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
  IndexInt load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
  // tid 需要去gmem上load的 行坐标
  if (load_gmem_Q_Br >= QKV_seqlen) return;
  // IndexInt load_gmem_K_Bc_offset = 0;

  // Dynamic shared memory allocation
  mem_alloc_shared_dynamic(fp16, smem);
  constexpr TileInt Q_tile_size = Br * kHeadDim;
  constexpr TileInt V_tile_size = Bc * kHeadDim;

  // Typed shared memory pointers
  SharedPtr<fp16> Q_tile_smem = shared_ptr_cast<fp16>(smem);
  SharedPtr<fp16> K_tile_smem = Q_tile_smem + Q_tile_size;
  SharedPtr<fp16> V_tile_smem = K_tile_smem;

  // Shared memory addresses for PTX instructions
  SmemAddr smem_Q_base_addr = generic_to_shared_addr(Q_tile_smem);
  SmemAddr smem_K_base_addr = generic_to_shared_addr(K_tile_smem);
  SmemAddr smem_V_base_addr = generic_to_shared_addr(V_tile_smem);

  /**
   * lane_block_row_max_old: reg_space<lane_row_max_old>(THREAD_NUM * 1 * kWarpTileSeqLenQ * 2)
   * lane_block_row_sum_old: reg_space<lane_row_sum_old>(THREAD_NUM * 1 * kWarpTileSeqLenQ * 2)
   */
  // Register memory allocation and initialization
  mem_alloc_register(fp32, lane_block_row_max_old, [kWarpTileSeqLenQ][2]); // [1][2]
  mem_fill(lane_block_row_max_old, -INFINITY);
  mem_alloc_register(fp32, lane_block_row_sum_old, [kWarpTileSeqLenQ][2]); // [1][2]
  mem_fill(lane_block_row_sum_old, 0.0f);

  /**
   * sizeof(uint32_t) / sizeof(fp16) = 2
   * R_Q: reg_space<THREAD_NUM, fp16, 2>[0 : kWarpTileSeqLenQ][0 : 4]
   * R_K: reg_space<THREAD_NUM, fp16, 2>[0 : kWarpTileSeqLenK][0 : 2]
   * R_V: reg_space<THREAD_NUM, fp16, 2>[0 : kWarpTileHeadDimV][0 : 2]
   * R_S: reg_space<THREAD_NUM, fp16, 2>[0 : kWarpTileSeqLenQ][0 : kWarpTileSeqLenK][0 : 2]
   * R_O: reg_space<THREAD_NUM, fp16, 2>[0 : kWarpTileSeqLenP][0 : kWarpTileHeadDimV][0 : 2]
   * R_D: reg_space<THREAD_NUM, fp16, 4>[0 : kWarpTileSeqLenP][0 : kWarpTileHeadDimV][0 : 4]
   */
  mem_alloc_register(uint32_t, R_Q, [kWarpTileSeqLenQ][4]); // [1][4]
  mem_alloc_register(uint32_t, R_K, [kWarpTileSeqLenK][2]); // [4][2]
  mem_alloc_register(uint32_t, R_V, [kWarpTileHeadDimV][2]); // [(kHeadDim / 8)][2]
  mem_alloc_register(uint32_t, R_S, [kWarpTileSeqLenQ][kWarpTileSeqLenK][2]); // [1][4][2]
  mem_alloc_register(uint32_t, R_O, [kWarpTileSeqLenP][kWarpTileHeadDimV][2]); // [1][(kHeadDim / 8)][2]
  mem_alloc_register(uint32_t, R_D, [kWarpTileSeqLenP][kWarpTileHeadDimV][4]); // [1][(kHeadDim / 8)][4]
  mem_fill(R_D, 0u);
  /** COPY: G2S
   * Q_tile_smem[0 : Br][0 : kHeadDim] = (Q + Q_gmem_offset)[Q_tile_id * Br : (Q_tile_id + 1) * Br][0 : kHeadDim]
   */
  // Initial async copy of Q from Global to Shared memory
  {
    IndexInt load_gmem_Q_d = load_smem_Q_d;
    IndexInt load_gmem_Q_addr = (Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
    // gmem_coord => (load_gmem_Q_Br, load_gmem_Q_d):(kHeadDim, 1)
    SmemAddr load_smem_Q_addr = smem_Q_base_addr + (load_smem_Q_Br * kHeadDim + load_smem_Q_d) * sizeof(fp16);
    // smem_coord => (load_smem_Q_Br, load_smem_Q_d):(kHeadDim, 1)
    serial_range_for(i, 0, (kHeadDim / (kNumThreads / Br)), 8) {
        warp_copy_async<fp16, 128, CopyTask::G2S>(&Q[load_gmem_Q_addr + i], load_smem_Q_addr + i * sizeof(fp16));
    }
    warp_copy_async_commit_group();
  }

  // Main loop over K/V tiles
  serial_range_for(tile_K_seqlen, 0, Tc, 1) {
    if (tile_K_seqlen == 0) {
      /** COPY G2S:
       * copy_shape: copy_block_tile_shape: (Bc, kHeadDim)
       * gmem_base_offset: K_gmem_offset
       * smem_base_offset: 0
       * block_coord: (tile_K_seqlen, 0); step_tile:(Bc, kHeadDim)
       * thread_copy_coord, in block_tile: 
       *   smem_coord: (load_smem_K_Bc, load_smem_K_d):(kHeadDim, 1):(0,0):(Bc, kHeadDim):smem_K_base_ptr
       *   gmem_coord: (load_gmem_K_Bc, load_gmem_K_d)=(load_smem_K_Bc, load_smem_K_d):(kHeadDim, 1):(tile_K_seqlen * Bc, 0):(QKV_seqlen, kHeadDim):&K[K_gmem_offset]
       * (crd_dim1, crd_dim2):(crd_stride_dim1, crd_stride_dim2):(start_offset_dim1, start_offset_dim2):(crd_space_dim1, crd_space_dim2)
       */
      // First iteration: load K tile and wait for Q and K
      // copy shape: (Bc, kHeadDim)
      // gmem_coord_stride = (kHeadDim, 1)
      // smem_coord_stride = (kHeadDim, 1) // 8
      /** COPY: G2S
       * K_tile_smem[0 : Bc][0 : kHeadDim] = (K + K_gmem_offset)[tile_K_seqlen * Bc : (tile_K_seqlen + 1) * Bc][0 : kHeadDim]
       */
      IndexInt load_gmem_K_Bc = tile_K_seqlen * Bc + load_smem_K_Bc; // thread_global_row_coord
      IndexInt load_gmem_K_d = load_smem_K_d; // thread_global_col_coord
      IndexInt load_gmem_K_idx_offset = K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d; // global_offset = global_base_offset + thread_global_row_coord * kHeadDim + thread_global_col_coord
      IndexInt load_smem_K_idx_offset = load_smem_K_Bc * kHeadDim + load_smem_K_d; // thread_smem_offset
      GlobalPtr<fp16> load_gmem_K_ptr = K + load_gmem_K_idx_offset;
      SharedPtr<fp16> load_smem_K_ptr = K_tile_smem + load_smem_K_idx_offset;
      serial_range_for(i, 0, (kHeadDim / (kNumThreads / Bc)), 8) {
        GlobalPtr<fp16> load_gmem_K_ptr_i = load_gmem_K_ptr + i;
        SharedPtr<fp16> load_smem_K_ptr_i = load_smem_K_ptr + i;
        SmemAddr load_smem_K_addr_i = generic_to_shared_addr(load_smem_K_ptr_i);
        warp_copy_async<fp16, 128, CopyTask::G2S>(load_gmem_K_ptr_i, load_smem_K_addr_i); // 8 ele
      }
      warp_copy_async_commit_group();
      warp_copy_async_wait_group<0>();
      block_sync();
    }
    
    // Load V tile for current iteration
    /** COPY: G2S
     * (V_tile_smem + V_tile_size)[0 : Bc][0 : kHeadDim] = (V + V_gmem_offset)[tile_K_seqlen * Bc : (tile_K_seqlen + 1) * Bc][0 : kHeadDim]
     */
    {
      /** COPY G2S:
        * copy_shape: copy_block_tile_shape: (Bc, kHeadDim)
        * gmem_base_offset: V_gmem_offset
        * smem_base_offset: smem_V_base_addr + V_tile_size * sizeof(half)
        * block_coord: (tile_K_seqlen, 0); step_tile:(Bc, kHeadDim)
        * thread_copy_coord, in block_tile: 
        *   smem_coord: (load_smem_V_Bc, load_smem_V_d):(kHeadDim, 1):(0,0):(Bc, kHeadDim):smem_V_base_ptr + V_tile_size * sizeof(half)
        *   gmem_coord: (load_gmem_V_Bc, load_gmem_V_d)=(load_smem_V_Bc, load_smem_V_d):(kHeadDim, 1):(tile_K_seqlen * Bc, 0):(QKV_seqlen, kHeadDim):&V[V_gmem_offset]
        * (crd_dim1, crd_dim2):(crd_stride_dim1, crd_stride_dim2):(start_offset_dim1, start_offset_dim2):(crd_space_dim1, crd_space_dim2)
        */
      IndexInt load_gmem_V_Bc = tile_K_seqlen * Bc + load_smem_V_Bc;
      IndexInt load_gmem_V_d = load_smem_V_d;
      IndexInt load_gmem_V_addr = V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d;
      SmemAddr load_smem_V_addr = smem_V_base_addr + (V_tile_size + load_smem_V_Bc * kHeadDim + load_smem_V_d) * sizeof(fp16);
      serial_range_for(i, 0, (kHeadDim / (kNumThreads / Bc)), 8) {
        warp_copy_async<fp16, 128, CopyTask::G2S>(&V[load_gmem_V_addr + i], load_smem_V_addr + i * sizeof(fp16));
      }
      warp_copy_async_commit_group();
    }

    // Initialize S accumulator registers
    mem_fill(R_S, 0u); // [1][4][2]

    /**
     * q_1 = Q_tile_smem[0 : Br][0 : kHeadDim]
     * k_1 = K_tile_smem[0 : Bc][0 : kHeadDim]
     * s_1 = q_1 @ k_1.T
     */
    /**
     * SOURCE:
     * <threads[0 : kNumThreads]><fp16, 2> R_Q[0 : kWarpTileSeqLenQ][0 : 4]
     * <threads[0 : kNumThreads]><fp16, 2> R_K[0 : kWarpTileSeqLenK][0 : 2]
     * TARGET:
     * <threads[0 : kNumThreads]><fp16, 2> R_S[0 : kWarpTileSeqLenQ][0 : kWarpTileSeqLenK][0 : 2]
     */
    // Inner loop for QK^T GEMM
    // 参与线程: $Thread[0 : kNumThreads] // [0:128]
    serial_range_for(tile_K_d, 0, (kHeadDim / kMmaAtomK), 1) { // d / 16
      // /**
      // * q_1_1 = Q_tile_smem[0 : Br][(tile_K_d * kMmaAtomK) : (tile_K_d + 1) * kMmaAtomK] => R_Q[0 : kWarpTileSeqLenQ][0 : 4]
      // * k_1_1 = K_tile_smem[0 : Bc][(tile_K_d * kMmaAtomK) : (tile_K_d + 1) * kMmaAtomK] => R_K[0 : kWarpTileSeqLenK][0 : 2]
      // * s_1_1: mem_alloc_tile[0 : Br][0 : Bc]
      // * s_1_1 = q_1 @ k_1_1.T => R_S[0 : kWarpTileSeqLenQ][0 : kWarpTileSeqLenK][0 : 2]
      // */

      /** COPY: R2S
       * q_1_1 = Q_tile_smem[0 : Br][(tile_K_d * kMmaAtomK) : (tile_K_d + 1) * kMmaAtomK] => R_Q[0 : kWarpTileSeqLenQ][0 : 4]
       */
      // Description: Load Q tile from Shared to Registers 
      serial_range_for(i, 0, kWarpTileSeqLenQ, 1) { // 1
        // q_1_1_1 = Q_tile_smem[0 : Br][(tile_K_d * kMmaAtomK) : (tile_K_d + 1) * kMmaAtomK] => R_Q[i][0 : 4]
        assert(kMmaAtomM == 16);
        assert(kMmaAtomK == 16);
        assert(Bc * kMmaAtomK * 1 == Bc * kMmaAtomK);
        IndexInt warp_smem_Q_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM; // warp_QP == warp_id
        IndexInt lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16; // lane在warp中的smem行坐标, lane本身排布列优先, 所以%16, 符合ldmatrix标准
        IndexInt lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8; // lane在warp中的smem列坐标
        SmemAddr lane_smem_Q_addr = smem_Q_base_addr + (lane_smem_Q_Br * kHeadDim + lane_smem_Q_d) * sizeof(fp16);
        warp_copy_sync<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(lane_smem_Q_addr, R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3]);
      }

      /**
       * Description: Load K tile from Shared to Registers
       * Annotation<Layout>: source row major in smem, target mma's col major in register, source a so load transpose row major in smem so naturally seen as col major in register space
       */
      /** COPY: R2S
       * k_1_1 = K_tile_smem[0 : Bc][(tile_K_d * kMmaAtomK) : (tile_K_d + 1) * kMmaAtomK]
       */
      /**
       * SOURCE:
       * K_tile_smem[0 : Bc][(tile_K_d * kMmaAtomK) : (tile_K_d + 1) * kMmaAtomK]
       * TARGET:
       * <threads[0 : kNumThreads]><fp16, 2> R_K[0 : kWarpTileSeqLenK][0 : 2]
       */
      // 循环长度 * 一次拷贝的总元素数量 = 4 * 512 = 2048
      // 实际访问数据元素个数: Bc * kMmaAtomK = 32 * 16 = 512
      // 不同warp持有相同数据, 多播次数: 4
      serial_range_for(j, 0, kWarpTileSeqLenK, 1) { // 4
        // 一个warp中32个线程, 只有前16个线程提供了地址，然而所有线程都收到了数据, 排布就是mma要求的B矩阵的layout https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-16816-b-f16
        // warp 0,1,2,3都将 K_tile_smem[j*kMmaAtomN : (j+1)*kMmaAtomN][(tile_K_d * kMmaAtomK) : (tile_K_d + 1) * kMmaAtomK]
        // 加载到自己线程的R_K[j][0 : 2]
        // kWarpTileSeqLenK 在后面的 mma中也是真正的迭代维度，每一次4个warp分别load的Q和同一个K矩阵乘，一个抽象的外积方式
        // 从线程发射角度考虑: 4个参与的warp, 每个warp有16个线程提供地址(有16个线程参与), 每个线程提供的地址负责拷贝64bit的元素(8个), 所以一次迭代拷贝的总元素数量: 4 * 16 * 8 = 512
        // 从线程获取结果空间角度考虑: 一次迭代拷贝的总元素数量 = 持有拷贝结果的线程数量 * (一次循环)持有拷贝结果的线程的存放结果的寄存器数量 * 持有拷贝结果的寄存器每个能放几个目标元素 = 128 * 2 * 2 = 512
        // 从数据空间被访问的角度考虑: 一次迭代数据涉及的smem的数据(坐标)个数: 8 * 16 = 128
        //                           因为不同warp访问相同地址造成的重叠4次, 从而一次迭代真正发送到register空间的数据个数: 128 * 4 = 512
        // assert(kMmaAtomN == 8);
        // assert(kMmaAtomK == 16);
        // const int repeat_num = warp_num;
        // // 源空间和目标空间的大小应该相等
        // assert(kMmaAtomN * kMmaAtomK * repeat_num == kNumThreads * 2 * 32/16);
        IndexInt warp_smem_K_Bc = j * kMmaAtomN; // j * 8
        IndexInt lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8;
        IndexInt lane_smem_K_d = tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8;
        SmemAddr lane_smem_K_addr = smem_K_base_addr + (lane_smem_K_Bc * kHeadDim + lane_smem_K_d) * sizeof(fp16);
        warp_copy_sync<fp16, 64, CopyTask::S2R, Layout::ROW_MAJOR>(lane_smem_K_addr, R_K[j][0], R_K[j][1]);
      }

      // Perform MMA for QK^T
      /**
       * s_1_1 = q_1_1 @ k_1_1.T
       */
      /** 
       * SOURCE:
       * q_1_1[0 : Br][0 : kMmaAtomK]: R_Q<[0 : THREAD_NUM],[0 : 4]>[0][0 : 4]
       * k_1_1[0 : Bc][0 : kMmaAtomK]: R_K<[0 :THREAD_NUM],[0 : 2]>[0][0 : 2]
       * TARGET:
       * s_1_1[0 : Br][0 : Bc]: R_S<[0 : THREAD_NUM],[0 : 2]>[0][0 : kWarpTileSeqLenK][0 : 2]
       * SPACE_CHECK
       * Br * Bc == THREAD_NUM * 2 * 1 * kWarpTileSeqLenK * 2)
       */
      serial_range_for(j, 0, kWarpTileSeqLenK, 1) { // 4
        mma<MmaShape::M16N8K16, MmaLayout::TN, fp16, fp16>(R_S[0][j][0], R_S[0][j][1], R_Q[0][0], R_Q[0][1], R_Q[0][2], R_Q[0][3], R_K[j][0], R_K[j][1], R_S[0][j][0], R_S[0][j][1]);
      }
    }
    block_sync();

    if ((tile_K_seqlen + 1) < Tc) {
      /**
      * COPY: G2S
      * Description: Load next K tile from Global Memory to Shared Memory asynchronously
      */
      /**
      * load_smem_K_Bc = range(0, Bc, 1) => [0 : Bc]
      * load_smem_K_d = range(0, kHeadDim, kHeadDim / (kNumThreads / Bc)) => [0 : kHeadDim : (kHeadDim / (kNumThreads / Bc))]
      */
      /** COPY: G2S
      * K_tile_smem[0 : Bc][0 : kHeadDim] = (K + K_gmem_offset)[(tile_K_seqlen + 1) * Bc : (tile_K_seqlen + 2) * Bc][0 : kHeadDim]
      */
      {
        IndexInt load_gmem_K_Bc_offset = (tile_K_seqlen + 1) * Bc;
        IndexInt load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
        IndexInt load_gmem_K_d = load_smem_K_d;
        IndexInt load_gmem_K_addr = (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
        SmemAddr load_smem_K_addr = smem_K_base_addr + (load_smem_K_Bc * kHeadDim + load_smem_K_d) * sizeof(fp16);
        serial_range_for(i, 0, (kHeadDim / (kNumThreads / Bc)), 8) {
          warp_copy_async<fp16, 128, CopyTask::G2S>(&K[load_gmem_K_addr + i], load_smem_K_addr + i * sizeof(fp16));
        }
        warp_copy_async_commit_group();
      }
    }

    // Softmax calculation
    /**
     * lane_row_max_new: <threads[0 : kNumThreads]><fp32>[kWarpTileSeqLenQ][2]
     * lane_row_sum_new: <threads[0 : kNumThreads]><fp32>[kWarpTileSeqLenQ][2]
     */
    mem_alloc_register(fp32, lane_row_max_new, [kWarpTileSeqLenQ][2]); // [1][2]
    mem_fill(lane_row_max_new, -INFINITY);
    mem_alloc_register(fp32, lane_row_sum_new, [kWarpTileSeqLenQ][2]); // [1][2]
    mem_fill(lane_row_sum_new, 0.0f);

    /**
     * row_max_1 = max(s_1, dim = -1, keepdim = False)
     */
    /**
     * s_1: R_S<THREAD_NUM, fp16, 2>[kWarpTileSeqLenK][2]
     * =>
     * row_max_1: lane_row_max_new<THREAD_NUM, fp32>[kWarpTileSeqLenQ][2]
     */
    {
      // Find local max in registers
      serial_range_for(j, 0, kWarpTileSeqLenK, 1) { // 4
        /**
         * row_max_1_1 = max(s_2_1, dim = -1, keepdim = False)
         */
        /**
         * s_2_1: R_S<THREAD_NUM, fp16, 2>[kWarpTileSeqLenK][2]
         * =>
         * row_max_1_1: lane_row_max_new<THREAD_NUM, fp32>[kWarpTileSeqLenQ][2]
         */
        fp16 *t_hptr_S_0_1 = register_as_fp16(&(R_S[0][j][0])); // RegisterPtr<fp16> t_hptr_S_0_1 = ptr_cast_register<fp16>(&R_S[0][j][0]);
        fp32 tmp_max_0 = compute_cast<fp16, fp32>(compute_max<fp16>(t_hptr_S_0_1[0], t_hptr_S_0_1[1])) * scale;
        fp32 tmp_max_1 = compute_cast<fp16, fp32>(compute_max<fp16>(t_hptr_S_0_1[2], t_hptr_S_0_1[3])) * scale;
        lane_row_max_new[0][0] = compute_max<fp32>(lane_row_max_new[0][0], tmp_max_0);
        lane_row_max_new[0][1] = compute_max<fp32>(lane_row_max_new[0][1], tmp_max_1);
      }

      // Sub-warp reduction to find max over 4 threads
      lane_row_max_new[0][0] = warp_shuffle_max_width<fp32>(lane_row_max_new[0][0], 4);
      lane_row_max_new[0][1] = warp_shuffle_max_width<fp32>(lane_row_max_new[0][1], 4);
    }

    /**
     * row_sum = sum(exp(s - row_max), dim = -1)
     * s = s / row_sum
     */
    {
      // Update global max and calculate probabilities
      fp32 block_row_max_new_0 = lane_row_max_new[0][0];
      fp32 block_row_max_new_1 = lane_row_max_new[0][1];
      fp32 block_row_max_old_0 = lane_block_row_max_old[0][0];
      fp32 block_row_max_old_1 = lane_block_row_max_old[0][1];
      block_row_max_new_0 = compute_max<fp32>(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = compute_max<fp32>(block_row_max_old_1, block_row_max_new_1);

      serial_range_for(j, 0, kWarpTileSeqLenK, 1) { // 4
        fp16 *t_hptr_S_0_1 = register_as_fp16(&(R_S[0][j][0]));
        fp32_4 t_reg_S_0_1;
        t_reg_S_0_1.x = compute_exp<fp32>(compute_fma<fp32>(compute_cast<fp16, fp32>(t_hptr_S_0_1[0]), scale, -block_row_max_new_0));
        t_reg_S_0_1.y = compute_exp<fp32>(compute_fma<fp32>(compute_cast<fp16, fp32>(t_hptr_S_0_1[1]), scale, -block_row_max_new_0));
        t_reg_S_0_1.z = compute_exp<fp32>(compute_fma<fp32>(compute_cast<fp16, fp32>(t_hptr_S_0_1[2]), scale, -block_row_max_new_1));
        t_reg_S_0_1.w = compute_exp<fp32>(compute_fma<fp32>(compute_cast<fp16, fp32>(t_hptr_S_0_1[3]), scale, -block_row_max_new_1));
        lane_row_sum_new[0][0] += (t_reg_S_0_1.x + t_reg_S_0_1.y);
        lane_row_sum_new[0][1] += (t_reg_S_0_1.z + t_reg_S_0_1.w);
        t_hptr_S_0_1[0] = compute_cast<fp32, fp16>(t_reg_S_0_1.x);
        t_hptr_S_0_1[1] = compute_cast<fp32, fp16>(t_reg_S_0_1.y);
        t_hptr_S_0_1[2] = compute_cast<fp32, fp16>(t_reg_S_0_1.z);
        t_hptr_S_0_1[3] = compute_cast<fp32, fp16>(t_reg_S_0_1.w);
      }

      // Sub-warp reduction for sum
      lane_row_sum_new[0][0] = warp_shuffle_sum_width<fp32>(lane_row_sum_new[0][0], 4);
      lane_row_sum_new[0][1] = warp_shuffle_sum_width<fp32>(lane_row_sum_new[0][1], 4);
    }

    // Wait for async copies to complete
    if ((tile_K_seqlen + 1) < Tc) {
      warp_copy_async_wait_group<1>();
    } else {
      warp_copy_async_wait_group<0>();
    }
    block_sync();

    // Initialize O accumulator registers
    mem_fill(R_O, 0u);
    
    // PV GEMM
    /**
     * O = O + s @ v.T
     */
    serial_range_for(tile_V_Bc, 0, (Bc / kMmaAtomK), 1) {
      // Load V tile (transposed) from Shared to Registers
      serial_range_for(j, 0, kWarpTileHeadDimV, 1) {
        IndexInt warp_smem_V_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
        IndexInt lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16;
        IndexInt lane_smem_V_d = warp_smem_V_d;
        SmemAddr lane_smem_V_addr = smem_V_base_addr + (V_tile_size + lane_smem_V_Bc * kHeadDim + lane_smem_V_d) * sizeof(fp16);
        warp_copy_sync<fp16, 64, CopyTask::S2R, Layout::COL_MAJOR>(lane_smem_V_addr, R_V[j][0], R_V[j][1]);
      }

      IndexInt w = tile_V_Bc * 2;
      static_assert(kWarpTileSeqLenP == 1);
      // Perform MMA for PV
      serial_range_for(j, 0, kWarpTileHeadDimV, 1) {
        mma<MmaShape::M16N8K16, MmaLayout::TN, fp16, fp16>(R_O[0][j][0], R_O[0][j][1], R_S[0][w][0], R_S[0][w][1], R_S[0][w + 1][0], R_S[0][w + 1][1], R_V[j][0], R_V[j][1], R_O[0][j][0], R_O[0][j][1]);
      }
    }
    block_sync();

    // Rescale accumulator O based on new max
    {
      fp32 block_row_max_new_0 = lane_row_max_new[0][0];
      fp32 block_row_max_new_1 = lane_row_max_new[0][1];
      fp32 block_row_max_old_0 = lane_block_row_max_old[0][0];
      fp32 block_row_max_old_1 = lane_block_row_max_old[0][1];
      block_row_max_new_0 = compute_max<fp32>(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = compute_max<fp32>(block_row_max_old_1, block_row_max_new_1);
      block_row_max_old_0 = (tile_K_seqlen > 0 ? block_row_max_old_0 : block_row_max_new_0);
      block_row_max_old_1 = (tile_K_seqlen > 0 ? block_row_max_old_1 : block_row_max_new_1);

      fp32 rescale_o_factor_0 = compute_exp<fp32>(block_row_max_old_0 - block_row_max_new_0);
      fp32 rescale_o_factor_1 = compute_exp<fp32>(block_row_max_old_1 - block_row_max_new_1);

      serial_range_for(j, 0, kWarpTileHeadDimV, 1) {
        fp16 *t_hptr_O_0_1 = register_as_fp16(&(R_O[0][j][0]));
        fp32 *t_fptr_D_0_1 = register_as_fp32(&(R_D[0][j][0]));
        t_fptr_D_0_1[0] = compute_rescale_accumulate<fp32>(t_fptr_D_0_1[0], compute_cast<fp16, fp32>(t_hptr_O_0_1[0]), rescale_o_factor_0);
        t_fptr_D_0_1[1] = compute_rescale_accumulate<fp32>(t_fptr_D_0_1[1], compute_cast<fp16, fp32>(t_hptr_O_0_1[1]), rescale_o_factor_0);
        t_fptr_D_0_1[2] = compute_rescale_accumulate<fp32>(t_fptr_D_0_1[2], compute_cast<fp16, fp32>(t_hptr_O_0_1[2]), rescale_o_factor_1);
        t_fptr_D_0_1[3] = compute_rescale_accumulate<fp32>(t_fptr_D_0_1[3], compute_cast<fp16, fp32>(t_hptr_O_0_1[3]), rescale_o_factor_1);
      }

      fp32 block_row_sum_old_0 = lane_block_row_sum_old[0][0];
      fp32 block_row_sum_old_1 = lane_block_row_sum_old[0][1];
      lane_block_row_sum_old[0][0] = compute_rescale_accumulate<fp32>(block_row_sum_old_0, lane_row_sum_new[0][0], rescale_o_factor_0);
      lane_block_row_sum_old[0][1] = compute_rescale_accumulate<fp32>(block_row_sum_old_1, lane_row_sum_new[0][1], rescale_o_factor_1);
      lane_block_row_max_old[0][0] = block_row_max_new_0;
      lane_block_row_max_old[0][1] = block_row_max_new_1;
    }

    // Wait for prefetched K tile if not the last iteration
    if ((tile_K_seqlen + 1) < Tc) {
      warp_copy_async_wait_group<0>();
      block_sync();
    }
  }
  block_sync();

  // Final normalization of O
  {
    fp32 rescale_factor_0 = compute_rcp<fp32>(lane_block_row_sum_old[0][0]);
    fp32 rescale_factor_1 = compute_rcp<fp32>(lane_block_row_sum_old[0][1]);
    serial_range_for(j, 0, kWarpTileHeadDimV, 1) {
      fp32 *t_fptr_D_0_1 = register_as_fp32(&(R_D[0][j][0]));
      fp16 *t_hptr_D_0_1 = register_as_fp16(&(R_D[0][j][0]));
      t_hptr_D_0_1[0] = compute_cast<fp32, fp16>(rescale_factor_0 * t_fptr_D_0_1[0]);
      t_hptr_D_0_1[1] = compute_cast<fp32, fp16>(rescale_factor_0 * t_fptr_D_0_1[1]);
      t_hptr_D_0_1[2] = compute_cast<fp32, fp16>(rescale_factor_1 * t_fptr_D_0_1[2]);
      t_hptr_D_0_1[3] = compute_cast<fp32, fp16>(rescale_factor_1 * t_fptr_D_0_1[3]);
    }
  }

  // Store final O from Registers to Global memory
  static_assert(kWarpTileSeqLenP == 1);
  {
    serial_range_for(j, 0, kWarpTileHeadDimV, 1) {
      mem_alloc_register(uint32_t, R_Z, [2][4]);
      // Redistribute results from 4 threads across the sub-warp
      warp_shuffle_spread_dual_x4(R_D[0][j][0], R_D[0][j][1], R_Z[0], R_Z[1], lane_id, 4);

      if (lane_id % 4 == 0) {
        IndexInt store_warp_regs_O_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + 0 * kMmaAtomM;
        IndexInt store_lane_gmem_O_Br = O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
        IndexInt store_warp_regs_O_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
        IndexInt store_lane_gmem_O_d = store_warp_regs_O_d;
        IndexInt store_gmem_O_addr_0 = (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim + store_lane_gmem_O_d);
        IndexInt store_gmem_O_addr_1 = (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim + store_lane_gmem_O_d);
        
        thread_copy_sync<fp32, 128, CopyTask::R2G>(&R_Z[0][0], &O[store_gmem_O_addr_0]);
        thread_copy_sync<fp32, 128, CopyTask::R2G>(&R_Z[1][0], &O[store_gmem_O_addr_1]);
      }
    }
  }
}

template <const TunableInt kHeadDim>
void launch_flash_attn_mma_stages_split_q_shared_kv(torch::Tensor Q, torch::Tensor K,
                                                   torch::Tensor V, torch::Tensor O) {
  constexpr TileInt kMmaAtomM = 16;
  constexpr TileInt kMmaAtomN = 8;
  constexpr TileInt kMmaAtomK = 16;
  constexpr TileInt kStage = 2;
  constexpr TileInt kMmaTileSeqLenQ = 4;
  constexpr TileInt kMmaTileSeqLenK = 1;
  constexpr TileInt kMmaTileSeqLenP = 4;
  constexpr TileInt kMmaTileHeadDimV = 1;
  constexpr TileInt kWarpTileSeqLenQ = 1;
  constexpr TileInt kWarpTileSeqLenK = 4;
  constexpr TileInt kWarpTileSeqLenP = 1;
  constexpr TileInt kWarpTileHeadDimV = (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV));
  constexpr TileInt Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr TileInt Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  constexpr TileInt kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  static_assert(kHeadDim < 256);

  constexpr TileInt Q_tile_size = (Br * kHeadDim);
  constexpr TileInt K_tile_size = (Bc * kHeadDim);
  constexpr TileInt V_tile_size = (Bc * kHeadDim);
  const TileInt smem_max_size = (Q_tile_size + kStage * max(K_tile_size, V_tile_size)) * sizeof(fp16);

  const ShapeInt QKV_batch = Q.size(0);
  const ShapeInt QKV_head = Q.size(1);
  const ShapeInt QKV_seqlen = Q.size(2);
  assert(QKV_seqlen % max(Br, Bc) == 0);
  const fp32 scale = 1.0f / sqrt((float)kHeadDim);

  dim3 grid(ceil_div(QKV_seqlen, Br), QKV_batch * QKV_head);
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
      <<<grid, block, smem_max_size>>>(
          (GlobalPtr<fp16>)Q.data_ptr(),
          (GlobalPtr<fp16>)K.data_ptr(),
          (GlobalPtr<fp16>)V.data_ptr(),
          (GlobalPtr<fp16>)O.data_ptr(),
          QKV_seqlen, QKV_head, scale);
}

void flash_attn_mma_stages_split_q_shared_kv(torch::Tensor Q, torch::Tensor K,
                                             torch::Tensor V, torch::Tensor O) {
  const ShapeInt d = Q.size(3);

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
      throw std::runtime_error("headdim not support!");
      break;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_attn_mma_stages_split_q_shared_kv",
    &flash_attn_mma_stages_split_q_shared_kv,
    "flash_attn_mma_stages_split_q_shared_kv");
}