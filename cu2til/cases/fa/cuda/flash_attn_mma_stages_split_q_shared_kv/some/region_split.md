**Segmenting the CUDA Kernel**

I'm focusing on segmenting the CUDA kernel code. The primary regions will be defined by loops and conditionals. I'll start by identifying the boundaries using these control flow statements to break it up into manageable parts.


**Decomposing Code Regions**

I'm currently categorizing the CUDA kernel into logical sections. Specifically, I've begun to establish regions like the Kernel Signature, setup, and Initial Data Loads. The `if` check is a good boundary. I'm also working on breaking the code up into Shared Memory and Register Setup.


**Refining Region Identification**

I'm now focusing on the detailed region breakdowns. The main loop structure seems to be the central point. Region 5, which is the Prefetch K/V block, needs to be divided. I've zeroed in on the asynchronous copy operations and synchronizations, especially how the `if (tile_K_seqlen == 0)` changes the prefetch behavior. This includes K and V loading.


**Detailing Kernel Regions**

I'm deep-diving into the exact composition of each region. Region 5, the K/V prefetch, is being further divided, with the first-iteration conditional taking special focus. Within the MMA loop (Region 7), I'm looking at breaking down the Q and K loads, along with the MMA itself, to create a detailed, segmented map of the kernel.


**Outlining Kernel Sections**

I'm now making a detailed structure based on control flow elements. This means identifying the `for` loops and `if` conditions. I'm starting with the Kernel Signature and the Initial Setup (Region 1). I'm also thinking about the shared memory and register setup to become Region 2, which is followed by the Initial Q Load as Region 3, and the Main Loop Start as Region 4. After this, it gets more complex.


**Outlining the CUDA Kernel**

I'm implementing the detailed segmentation, and I've started mapping regions. I'm focusing on control flow like `for` loops and `if` conditions to define blocks. Region 1 is the Kernel Signature and Initial Setup, including all the setup logic until the first major control flow.


**Detailing the Code Sections**

I'm now constructing a detailed map of the CUDA kernel, and it's becoming clearer. I have finished the initial segmentation, starting with the Kernel Signature (Region 1). Region 2 is the Shared Memory and Register Setup, with an initial Q load in Region 3. I've now identified the Main Loop Start as Region 4. I have also begun detailing the structure of the Main Loop, with Regions 5 through 17.


**Finalizing the Segmentation Plan**

I'm finalizing the detailed regions of the CUDA kernel. I've refined the breakdown, including each block and where the control flow starts and ends. Now, I'm preparing for code splitting, ensuring no modifications to the original code itself. I am double checking that no new characters are added. I will create the `cpp` blocks as required.


**Confirming Code Segmentation**

I've reviewed the division plan thoroughly, refining regions and synchronization points. All regions are well-defined, with loop boundaries and synchronization primitives acting as markers. My focus has shifted to code generation, making sure no modifications are made to the original code and that `cpp` blocks are utilized correctly.


**Confirming the Plan**

I've finalized the segmentation plan with clear region definitions. Loops, conditionals, and synchronization points have been considered for accurate mapping. Now, I will focus on implementing this plan, taking great care to preserve the original code structure, and using `cpp` blocks for separation.


Based on the control flow and synchronization statements, the provided CUDA kernel can be divided into the following regions:

```cpp
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
  const int Tc = ((QKV_seqlen % Bc != 0) ? (QKV_seqlen / Bc + 1) : (QKV_seqlen / Bc));
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
```

```cpp
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
  {
#pragma unroll
    for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        lane_block_row_max_old[i][j] = -INFINITY;
      }
    }
  }
  {
#pragma unroll
    for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        lane_block_row_sum_old[i][j] = 0.0f;
      }
    }
  }
```

```cpp
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
  {
#pragma unroll
    for (int i = 0; i < kWarpTileSeqLenP; ++i) {
#pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
#pragma unroll
        for (int k = 0; k < ((kOStorageAccFloat32) ? 4 : 2); ++k) {
          R_D[i][j][k] = 0;
        }
      }
    }
  }
```

```cpp
  {
    int load_gmem_Q_d = load_smem_Q_d;
    int load_gmem_Q_addr =
        (Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
    uint32_t load_smem_Q_ptr =
        (smem_Q_base_ptr +
         (load_smem_Q_Br * (kHeadDim + kPadQ) + load_smem_Q_d) * sizeof(half));
#pragma unroll
    for (int i = 0; i < (kHeadDim / (kNumThreads / Br)); i += 8) {
      asm volatile( 
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(load_smem_Q_ptr + i * 2), 
      "l"(&Q[load_gmem_Q_addr + i]), "n"(16));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  }
```

```cpp
#pragma unroll 1
  for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; ++tile_K_seqlen) {
```

```cpp
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
          asm volatile( 
              "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(load_smem_K_ptr + i * 2), 
              "l"(&K[load_gmem_K_addr + i]), "n"(16));
        }
        asm volatile("cp.async.commit_group;\n" ::);

        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();
      }
```

```cpp
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
          asm volatile( 
              "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(load_smem_V_ptr + i * 2), 
              "l"(&V[load_gmem_V_addr + i]), "n"(16));
        }
        asm volatile("cp.async.commit_group;\n" ::);
      }
    } else {
```

```cpp
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
        asm volatile( 
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(load_smem_K_ptr + i * 2), 
            "l"(&K[load_gmem_K_addr + i]), "n"(16));
      }
      asm volatile("cp.async.commit_group;\n" ::);
      asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
      __syncthreads();
    }
```

```cpp
    if constexpr (kCanPrefetchQs2r && (!kDelayPrefetchQs2r)) {
      if (tile_K_seqlen == 0) {
        if constexpr (!kCanPrefetchKVg2s) {
          asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        } else {
          asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
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
            asm volatile( 
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" 
                : "=r"(R_Q[tile_K_d][i][0]), "=r"(R_Q[tile_K_d][i][1]), "=r"(R_Q[tile_K_d][i][2]), "=r"(R_Q[tile_K_d][i][3]) 
                : "r"(lane_smem_Q_ptr));
          }
        }
        __syncthreads();
      }
    }
```

```cpp
    {
#pragma unroll
      for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpTileSeqLenK; ++j) {
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            R_S[i][j][k] = 0;
          }
        }
      }
    }
```

```cpp
#pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
```

```cpp
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
          asm volatile( 
              "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" 
              : "=r"(R_Q[0][i][0]), "=r"(R_Q[0][i][1]), "=r"(R_Q[0][i][2]), "=r"(R_Q[0][i][3]) 
              : "r"(lane_smem_Q_ptr));
        }
      } else {
        if constexpr (kDelayPrefetchQs2r) {
          if (tile_K_seqlen == 0) {
            if (tile_K_d == 0) {
              if constexpr (!kCanPrefetchKVg2s) {
                asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
              } else {
                asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
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
              asm volatile( 
                  "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" 
                  : "=r"(R_Q[tile_K_d][i][0]), "=r"(R_Q[tile_K_d][i][1]), "=r"(R_Q[tile_K_d][i][2]), "=r"(R_Q[tile_K_d][i][3]) 
                  : "r"(lane_smem_Q_ptr));
            }
          }
        }
      }
```

```cpp
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
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" 
                   : "=r"(R_K[j][0]), "=r"(R_K[j][1]) 
                   : "r"(lane_smem_K_ptr));
      }
```

```cpp
      if constexpr (kCanPrefetchQs2r) {
        static_assert(kWarpTileSeqLenQ == 1);
        {
#pragma unroll
          for (int j = 0; j < kWarpTileSeqLenK; ++j) {
            asm volatile( 
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, " 
                "%4, %5}, {%6, %7}, {%8, %9};\n" 
                : "=r"(R_S[0][j][0]), "=r"(R_S[0][j][1]) 
                : "r"(R_Q[tile_K_d][0][0]), "r"(R_Q[tile_K_d][0][1]), "r"(R_Q[tile_K_d][0][2]), "r"(R_Q[tile_K_d][0][3]), "r"(R_K[j][0]), "r"(R_K[j][1]), "r"(R_S[0][j][0]), 
                  "r"(R_S[0][j][1]));
          }
        }
      } else {
        static_assert(kWarpTileSeqLenQ == 1);
        {
#pragma unroll
          for (int j = 0; j < kWarpTileSeqLenK; ++j) {
            asm volatile( 
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, " 
                "%4, %5}, {%6, %7}, {%8, %9};\n" 
                : "=r"(R_S[0][j][0]), "=r"(R_S[0][j][1]) 
                : "r"(R_Q[0][0][0]), "r"(R_Q[0][0][1]), "r"(R_Q[0][0][2]), "r"(R_Q[0][0][3]), "r"(R_K[j][0]), "r"(R_K[j][1]), "r"(R_S[0][j][0]), 
                  "r"(R_S[0][j][1]));
          }
        }
      }
    }
    __syncthreads();
```

```cpp
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
        asm volatile( 
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(load_smem_V_ptr + i * 2), 
            "l"(&V[load_gmem_V_addr + i]), "n"(16));
      }
      asm volatile("cp.async.commit_group;\n" ::);
    }
```

```cpp
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
          asm volatile( 
              "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(load_smem_K_ptr + i * 2), 
              "l"(&K[load_gmem_K_addr + i]), "n"(16));
        }
        asm volatile("cp.async.commit_group;\n" ::);
      }
    }
```

```cpp
    float lane_row_max_new[kWarpTileSeqLenQ][2];
    float lane_row_sum_new[kWarpTileSeqLenQ][2];
    {
#pragma unroll
      for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          lane_row_max_new[i][j] = -INFINITY;
        }
      }
    }
    {
#pragma unroll
      for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          lane_row_sum_new[i][j] = 0.0f;
        }
      }
    }
```

```cpp
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

      {
        float _inlined_val_ = lane_row_max_new[0][0];
#pragma unroll
        for (int mask = 4 >> 1; mask >= 1; mask >>= 1) {
          _inlined_val_ = max(_inlined_val_, __shfl_xor_sync(0xffffffff, _inlined_val_, mask, 4));
        }
        lane_row_max_new[0][0] = _inlined_val_;
      }
      {
        float _inlined_val_ = lane_row_max_new[0][1];
#pragma unroll
        for (int mask = 4 >> 1; mask >= 1; mask >>= 1) {
          _inlined_val_ = max(_inlined_val_, __shfl_xor_sync(0xffffffff, _inlined_val_, mask, 4));
        }
        lane_row_max_new[0][1] = _inlined_val_;
      }
    }
```

```cpp
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
        float _inlined_val_ = lane_row_sum_new[0][0];
#pragma unroll
        for (int mask = 4 >> 1; mask >= 1; mask >>= 1) {
          _inlined_val_ += __shfl_xor_sync(0xffffffff, _inlined_val_, mask, 4);
        }
        lane_row_sum_new[0][0] = _inlined_val_;
      }
      {
        float _inlined_val_ = lane_row_sum_new[0][1];
#pragma unroll
        for (int mask = 4 >> 1; mask >= 1; mask >>= 1) {
          _inlined_val_ += __shfl_xor_sync(0xffffffff, _inlined_val_, mask, 4);
        }
        lane_row_sum_new[0][1] = _inlined_val_;
      }
    }
```

```cpp
    if constexpr (kCanPrefetchKVg2s) {
      if ((tile_K_seqlen + 1) < Tc) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
      } else {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
      }
    } else {
      asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    }
    __syncthreads();
```

```cpp
    {
#pragma unroll
      for (int i = 0; i < kWarpTileSeqLenP; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpTileHeadDimV; ++j) {
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            R_O[i][j][k] = 0;
          }
        }
      }
    }
```

```cpp
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
        asm volatile( 
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" 
            : "=r"(R_V[j][0]), "=r"(R_V[j][1]) 
            : "r"(lane_smem_V_ptr));
      }

      int w = tile_V_Bc * 2;
      static_assert(kWarpTileSeqLenP == 1);
      {
#pragma unroll
        for (int j = 0; j < kWarpTileHeadDimV; ++j) {
          asm volatile( 
              "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, " 
              "%4, %5}, {%6, %7}, {%8, %9};\n" 
              : "=r"(R_O[0][j][0]), "=r"(R_O[0][j][1]) 
              : "r"(R_S[0][w][0]), "r"(R_S[0][w][1]), "r"(R_S[0][w + 1][0]), "r"(R_S[0][w + 1][1]), "r"(R_V[j][0]), "r"(R_V[j][1]), "r"(R_O[0][j][0]), 
                "r"(R_O[0][j][1]));
        }
      }
    }
    __syncthreads();
```

```cpp
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
```

```cpp
    if constexpr (kCanPrefetchKVg2s) {
      if ((tile_K_seqlen + 1) < Tc) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();
      }
    }
  }
  __syncthreads();
```

```cpp
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
```

```cpp
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
          (reinterpret_cast<float4 *>(&(O[store_gmem_O_addr_0]))[0]) = ((reinterpret_cast<float4 *>(&(R_Q[0][0][0]))[0]));
          (reinterpret_cast<float4 *>(&(O[store_gmem_O_addr_1]))[0]) = ((reinterpret_cast<float4 *>(&(R_Q[1][0][0]))[0]));
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
          (reinterpret_cast<float4 *>(&(O[store_gmem_O_addr_0]))[0]) = ((reinterpret_cast<float4 *>(&(R_Z[0][0]))[0]));
          (reinterpret_cast<float4 *>(&(O[store_gmem_O_addr_1]))[0]) = ((reinterpret_cast<float4 *>(&(R_Z[1][0]))[0]));
        }
      }
    }
  }
}
```