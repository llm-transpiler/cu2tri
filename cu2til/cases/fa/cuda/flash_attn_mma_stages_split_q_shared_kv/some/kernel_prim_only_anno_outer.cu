
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
  constexpr TileInt Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr TileInt Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  constexpr TileInt kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  const IndexInt Tc = QKV_seqlen / Bc;
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
  IndexInt load_smem_Q_Br = (tid / (kNumThreads / Br));
  IndexInt load_smem_Q_d = (tid % (kNumThreads / Br)) * (kHeadDim / (kNumThreads / Br));
  IndexInt load_smem_K_Bc = (tid / (kNumThreads / Bc));
  IndexInt load_smem_K_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc))
  IndexInt load_smem_V_Bc = (tid / (kNumThreads / Bc));
  IndexInt load_smem_V_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
  IndexInt load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
  
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
   * lane_block_row_max_old: reg_space<threads[0 : THREAD_NUM]><fp32, 1>[kWarpTileSeqLenQ][2]
   * lane_block_row_sum_old: reg_space<threads[0 : THREAD_NUM]><fp32, 1>[kWarpTileSeqLenQ][2]
   */
  mem_alloc_register(fp32, lane_block_row_max_old, [kWarpTileSeqLenQ][2]); // [1][2]
  mem_fill(lane_block_row_max_old, -INFINITY);
  mem_alloc_register(fp32, lane_block_row_sum_old, [kWarpTileSeqLenQ][2]); // [1][2]
  mem_fill(lane_block_row_sum_old, 0.0f);

  /**
   * R_Q: alloc_reg_space<[0 : THREAD_NUM]><fp16, [0:2]>[0 : kWarpTileSeqLenQ][0 : 4]
   * R_K: alloc_reg_space<[0 : THREAD_NUM]><fp16, [0:2]>[0 : kWarpTileSeqLenK][0 : 2]
   * R_V: alloc_reg_space<[0 : THREAD_NUM]><fp16, [0:2]>[0 : kWarpTileHeadDimV][0 : 2]
   * R_S: alloc_reg_space<[0 : THREAD_NUM]><fp16, [0:2]>[0 : kWarpTileSeqLenQ][0 : kWarpTileSeqLenK][0 : 2]
   * R_O: alloc_reg_space<[0 : THREAD_NUM]><fp16, [0:2]>[0 : kWarpTileSeqLenP][0 : kWarpTileHeadDimV][0 : 2]
   * R_D: alloc_reg_space<[0 : THREAD_NUM]><fp16, [0:2]>[0 : kWarpTileSeqLenP][0 : kWarpTileHeadDimV][0 : 4]
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
  // {...}
  serial_range_for(tile_K_seqlen, 0, Tc, 1) {
    if (tile_K_seqlen == 0) {
      /** COPY: G2S
       * K_tile_smem[0 : Bc][0 : kHeadDim] = (K + K_gmem_offset)[tile_K_seqlen * Bc : (tile_K_seqlen + 1) * Bc][0 : kHeadDim]
       */
      // {...}
    }
    /** COPY: G2S
     * (V_tile_smem + V_tile_size)[0 : Bc][0 : kHeadDim] = (V + V_gmem_offset)[tile_K_seqlen * Bc : (tile_K_seqlen + 1) * Bc][0 : kHeadDim]
     */
    // {...}
    mem_fill(R_S, 0u);
    
    /**
     * q_1 = Q_tile_smem[0 : Br][0 : kHeadDim]
     * k_1 = K_tile_smem[0 : Bc][0 : kHeadDim]
     * s_1 = q_1 @ k_1.T
     * SOURCE:
     * Q_tile_smem[0 : Br][0 : kHeadDim]
     * K_tile_smem[0 : Bc][0 : kHeadDim]
     * TARGET:
     * s_1[0 : Br][0 : Bc]: R_S<[0 : THREAD_NUM]><fp16, [0:2]>[0 : kWarpTileSeqLenQ][0 : kWarpTileSeqLenK][0 : 2]
     * CHECK:
     * ITER: kHeadDim / kMmaAtomK * kMmaAtomK == kHeadDim
     * SPACE: Br * Bc == THREAD_NUM * 2 * 1 * kWarpTileSeqLenK * 2
     */
    // {...}
    if ((tile_K_seqlen + 1) < Tc) {
      /** COPY: G2S
       * K_tile_smem[0 : Bc][0 : kHeadDim] = (K + K_gmem_offset)[(tile_K_seqlen + 1) * Bc : (tile_K_seqlen + 2) * Bc][0 : kHeadDim]
       * SOURCE:
       * (K + K_gmem_offset)[(tile_K_seqlen + 1) * Bc : (tile_K_seqlen + 2) * Bc][0 : kHeadDim]
       * TARGET:
       * K_tile_smem[0 : Bc][0 : kHeadDim]
       * CHECK:
       */
    }
    /**
     * lane_row_max_new: reg_space<[0 : THREAD_NUM]><fp32, [0:1]>[kWarpTileSeqLenQ][2]
     * fill(lane_row_max, -inf)
     * lane_row_sum_new: reg_space<[0 : THREAD_NUM]><fp32, [0:1]>[kWarpTileSeqLenQ][2]
     * fill(lane_row_sum, 0.0)
     */
    mem_alloc_register(fp32, lane_row_max_new, [kWarpTileSeqLenQ][2]); // [1][2]
    mem_fill(lane_row_max_new, -INFINITY);
    mem_alloc_register(fp32, lane_row_sum_new, [kWarpTileSeqLenQ][2]); // [1][2]
    mem_fill(lane_row_sum_new, 0.0f);
    /** COMPUTE: CAST, MAX, MUL
     * EXPR:
     * row_max_1 = max(row_max_1, s_1.max(dim = -1) * scale, dim = -1)
     * SOURCE:
     * s_1[0 : Br][0 : Bc]: R_S<[0 : THREAD_NUM]><fp16, [0:2]>[0 : kWarpTileSeqLenK][0 : 2]
     * row_max_1[0 : Br][0 : Bc / 4 / 2]: lane_row_max_new<[0 : THREAD_NUM]><fp32, [0:1]>[0][2]
     * TARGET:
     * row_max_1[0 : Br][0 : Bc / 4 / 2]: lane_row_max_new<[0 : THREAD_NUM]><fp32, [0:1]>[0][2]
     * CHECK:
     * ITER: kWarpTileSeqLenK * kMmaAtomN / 2 == Bc / 2
     * TARGET SPACE: Br * Bc / 4 / 2 = THREAD_NUM * 1 * 2
     * 只需要查TARGETSPACE因为SOURCE SPACE一定是之前某个TARGET SPACE，不一定！！
     */
    // {...}
    /** COMPUTE: SUM, EXP, FMA, CAST, MAX
     * EXPR:
     * row_sum_1 = exp(s_1.to(fp32) * scale - max(block_row_max_old, row_max_1, dim = -1)).sum(dim = -1)
     * SOURCE:
     * s_1[0 : Br][0 : Bc]: R_S<[0 : THREAD_NUM]><fp16, [0:2]>[0 : kWarpTileSeqLenK][0 : 2]
     * scale: scalar<fp32>
     * block_row_max_old[0 : Br][0 : kMmaAtomN / 2]: lane_block_row_max_old<[0 : THREAD_NUM]><fp32, [0:1]>[0 : 2]
     * row_max_1[0 : Br][0 : kMmaAtomN / 2]: lane_row_max_new<[0 : THREAD_NUM]><fp32, [0:1]>[0 : 2]
     * TARGET:
     * row_sum_1[0 : Br][0 : kMmaAtomN / 2]: lane_row_sum_new<[0 : THREAD_NUM]><fp32, [0:1]>[0 : 2]
     * CHECK:
     * TARGET SPACE: Br * kMmaAtomN / 2 = THREAD_NUM * 1 * 2
     */
    // {...}
    if ((tile_K_seqlen + 1) < Tc) {
      warp_copy_async_wait_group<1>();
    } else {
      warp_copy_async_wait_group<0>();
    }
    block_sync();
    mem_fill(R_O, 0u);
    /** COMPUTE: GEMM: (Br, kHeadDim) = (Br, Bc) @ (Bc, kHeadDim)
     * EXPR:
     * o_1 = s_1 @ v_1
     * SOURCE:
     * s_1[0 : Br][0 : Bc]: R_S<[0 : THREAD_NUM]><fp16, [0:2]>[0 : kWarpTileSeqLenK][0 : 2]
     * v_1[0 : Bc][0 : kHeadDim]: R_V<[0 : THREAD_NUM]><fp16, [0:2]>[0 : kWarpTileHeadDimV][0 : 2]
     * TARGET:
     * o_1[0 : Br][0 : kHeadDim]: R_O<[0 : THREAD_NUM]><fp16, [0:2]>[0][0 : kWarpTileHeadDimV][0 : 2]
     * CHECK:
     * ITER: Bc == kMmaAtomK * (Bc / kMmaAtomK)
     * TARGET SPACE: Br * kHeadDim == THREAD_NUM * 2 * kWarpTileHeadDimV * 2
     */
    // {...}
    
    /**
     * EXPR:
     * rescale_o_factor = exp(block_row_max_old - row_max_1)
     * d_1[0 : Br][0 : kHeadDim] = d_1[0 : Br][0 : kMmaAtomN] * rescale_o_factor[0 : Br][0 : kHeadDim]
     */
    // {...}
    if ((tile_K_seqlen + 1) < Tc) {
      warp_copy_async_wait_group<0>();
      block_sync();
    }
  }
  /** COMPUTE: RCP, MUL, CAST
   * EXPR:
   * d[0 : Br][0 : kHeadDim] = (d[0 : Br][0 : kHeadDim] * 1 / block_row_sum_old[0 : Br][0 : kHeadDim]).to(fp16)
   */
  // {...}
  /**
   * EXPR:
   * (O + O_gmem_offset)[0 : Br][0 : kHeadDim] = d[0 : Br][0 : kHeadDim]
   */
  // {...}
}