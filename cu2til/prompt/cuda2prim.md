# CUDA Kernel to Tile-Level DSL Conversion Prompt

## Task
Convert optimized CUDA kernel code to use tile-level DSL primitives for better abstraction and code generation. The conversion should maintain identical functionality while using standardized primitives.

## Background
You are given a highly optimized CUDA kernel that uses direct PTX instructions and low-level CUDA operations. Your task is to convert it to use a set of predefined tile-level DSL primitives that abstract common GPU compute patterns.

## Input Format
- Original CUDA kernel file using direct PTX instructions and raw CUDA operations
- Assumes you have access to the dsl_template.cuh header with all primitive definitions

## Output Format  
- Converted CUDA kernel using only the provided DSL primitives
- Must include the dsl_template.cuh header
- All low-level operations replaced with corresponding DSL primitives

## Conversion Rules

### 1. Header and Includes
```cpp
// Add at the top after other includes:
#include "dsl_template.cuh"
```

### 2. Type Declarations
Replace C++ types with semantic DSL type aliases based on the **purpose** and **usage context**:

#### **TunableInt**
Use for kernel template parameters that can be tuned/configured:
```cpp
// FROM:
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int TILE_SIZE = 64, const int BLOCK_THREADS = 256>

// TO:  
template <const TunableInt MMA_M = 16, const TunableInt MMA_N = 8, const TunableInt MMA_K = 16,
          const TunableInt TILE_SIZE = 64, const TunableInt BLOCK_THREADS = 256>
```
**When to use**: Template parameters, configurable constants, padding values, any parameter that might be tuned for different architectures or use cases.

#### **ShapeInt** 
Use for dynamic tensor/matrix dimensions passed at runtime:
```cpp
// FROM:
__global__ void kernel(float* A, float* B, int M, int N, int K)

// TO:
__global__ void kernel(float* A, float* B, ShapeInt M, ShapeInt N, ShapeInt K)
```
**When to use**: Matrix dimensions (M, N, K), tensor sizes, dynamic array lengths, input/output dimensions.

#### **IndexInt**
Use for loop variables, array indices, and runtime calculated values:
```cpp
// FROM:
for (int i = 0; i < M; ++i) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int global_idx = i * N + offset;
}

// TO:
for (IndexInt i = 0; i < M; ++i) {
    IndexInt offset = blockIdx.x * blockDim.x + threadIdx.x;
    IndexInt global_idx = i * N + offset;
}
```
**When to use**: Loop counters, thread/block indices, address calculations, any runtime computed integer values.

#### **TileInt**
Use for computed tile/block dimensions (typically products of tunable parameters):
```cpp
// FROM:
const int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;  // Block M dimension
const int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;  // Block N dimension

// TO:
constexpr TileInt BM = MMA_M * MMA_TILE_M * WARP_TILE_M;
constexpr TileInt BN = MMA_N * MMA_TILE_N * WARP_TILE_N;
```
**When to use**: Computed tile sizes, block dimensions, shared memory dimensions derived from tunable parameters.

#### **SmemAddr**
Use **ONLY** for shared memory addresses in copy instructions:
```cpp
// FROM:
uint32_t smem_addr = __cvta_generic_to_shared(&shared_mem[0]);

// TO:  
SmemAddr smem_addr = generic_to_shared_addr(&shared_mem[0]);
```
**When to use**: ONLY for shared memory addresses used in PTX instructions (ldmatrix, cp.async, etc.) - either direct `__cvta_generic_to_shared` return values or addresses computed from them.

#### **Memory Space-Aware Pointer Types**
Use semantic pointer types to distinguish different memory spaces:
```cpp
// Global memory pointers
GlobalPtr<fp16> g_A = A;
GlobalPtr<fp32> g_B = B;

// Shared memory pointers  
SharedPtr<fp16> s_tile = &shared_array[0];
SharedPtr<fp32> s_data = shared_ptr_cast<fp32>(smem);

// Register arrays (conceptual - still regular arrays)  
RegisterPtr<uint32_t> reg_data = register_array;
```
**When to use**: 
- `GlobalPtr<T>` - Global memory parameters and pointers
- `SharedPtr<T>` - Static/dynamic shared memory pointers  
- `RegisterPtr<T>` - Register array access (conceptual clarity)
- **NOT for SmemAddr** - Keep SmemAddr only for copy instruction addresses

**Critical Distinction - Memory Pointer Types vs SmemAddr:**
```cpp
// ❌ WRONG - Don't use SmemAddr for regular memory access
SmemAddr bad_addr = some_array;  // NO! SmemAddr is for PTX instructions only

// ✅ CORRECT - Use semantic pointer types for memory access  
SharedPtr<fp16> good_shared_ptr = shared_array;       // Regular shared memory access
GlobalPtr<fp32> good_global_ptr = global_array;       // Global memory parameters
SmemAddr ptx_addr = generic_to_shared_addr(shared_array); // For PTX instructions (cp.async, ldmatrix)
```

**Memory Access Pattern:**
1. **Use `SharedPtr<T>` / `GlobalPtr<T>`** → Regular C++ operations, array indexing, pointer arithmetic
2. **Use `SmemAddr`** → PTX instructions requiring shared memory addresses (cp.async, ldmatrix, stmatrix)
3. **Convert at call site** → Use `generic_to_shared_addr()` when calling warp copy functions

**Simplified Decision Rule:**
**Will this address be used in PTX instruction (cp.async, ldmatrix, stmatrix)?** → Use `SmemAddr`. **Everything else?** → Use `SharedPtr<T>` / `GlobalPtr<T>`

**Memory Address Usage Examples:**
```cpp
// ✅ CORRECT - SharedPtr for regular operations
SharedPtr<fp16> s_a = shared_ptr_cast<fp16>(smem);
SharedPtr<fp16> lane_ptr = s_a + (m * stride + k);  // Pointer arithmetic

// ✅ CORRECT - SmemAddr for async copy instructions
SmemAddr smem_base_addr = generic_to_shared_addr(s_a);
SmemAddr load_addr = smem_base_addr + (m * stride + k) * sizeof(fp16);
warp_copy_async<fp16, 128, CopyTask::G2S>(&A[addr], load_addr);

// ✅ CORRECT - SmemAddr for sync warp copy operations
SmemAddr matrix_addr = generic_to_shared_addr(lane_ptr);
warp_copy_sync<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(matrix_addr, reg0, reg1, reg2, reg3);
```

#### **Predefined Data Types**
Use predefined type aliases for better portability and clarity:
```cpp
// 16-bit floating-point types
using fp16 = half;
using fp16_2 = half2;
using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

// 32-bit floating-point types  
using fp32 = float;
using fp32_2 = float2;
using fp32_4 = float4;

// Integer types
using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
```

#### **Architecture Constants**
Use predefined macros from dsl_template.cuh instead of defining your own:
```cpp
// FROM:
const int WARP_SIZE = 32;
int warp_id = tid / WARP_SIZE;

// TO:
IndexInt warp_id = tid / WARP_SIZE;  // Use predefined WARP_SIZE macro
```
**Available constants**: `WARP_SIZE`, `MAX_THREADS_PER_BLOCK`, `MAX_SHARED_MEM_PER_BLOCK`, etc.

### 3. Memory Allocation
Replace direct memory declarations:

**Shared Memory (Static):**
```cpp
// FROM:
__shared__ dtype var_name[dim1][dim2];

// TO:
mem_alloc_shared(dtype, var_name, [dim1][dim2]);
```

**Shared Memory (Dynamic):**
```cpp
// FROM:
extern __shared__ dtype var_name[];

// TO:
mem_alloc_shared_dynamic(dtype, var_name);
```

Note: Dynamic shared memory size is specified at kernel launch time via the third parameter of the kernel launch configuration (`<<<grid, block, shared_mem_size>>>`). The DSL primitive only handles the declaration; the size is still controlled by the launch parameters.

**Register Memory:**
```cpp
// FROM:
dtype var_name[size];
dtype var_name[size] = {init_values};

// TO:
mem_alloc_register(dtype, var_name, [size]);
// If initialization needed:
mem_fill(var_name, init_value);
```

### 4. Memory Copy Operations

**Memory Copy Operation Hierarchy:**

The DSL provides a clear three-tier hierarchy for memory operations, each optimized for different use cases:

1. **Thread-Level Operations (`thread_copy_sync`)**: 
   - **Purpose**: Simple, direct memory transfers within a single thread
   - **Use Cases**: Basic data movement, element-wise operations
   - **Memory Spaces**: All combinations (G2S, S2G, G2R, R2G, S2R, R2S)
   - **Synchronization**: Immediate, thread-local

2. **Warp-Level Async Operations (`warp_copy_async`)**: 
   - **Purpose**: High-throughput memory pipeline using CP.ASYNC hardware
   - **Use Cases**: Global → Shared memory bulk transfers ONLY
   - **Memory Spaces**: G2S only (CP.ASYNC hardware limitation)
   - **Address Requirements**: `SmemAddr` for shared memory destination
   - **Synchronization**: Asynchronous with explicit commit/wait groups
   - **Hardware**: Direct CP.ASYNC PTX instructions
   - **Note**: For Shared → Global, use `thread_copy_sync` instead

3. **Warp-Level Sync Operations (`warp_copy_sync`)**: 
   - **Purpose**: Matrix fragment operations using LDMATRIX/STMATRIX hardware
   - **Use Cases**: Loading matrix tiles for tensor core MMA operations  
   - **Memory Spaces**: S2R, R2S (optimized for matrix fragments)
   - **Address Requirements**: Requires `SmemAddr` for shared memory access
   - **Synchronization**: Warp-synchronous execution
   - **Hardware**: Direct LDMATRIX/STMATRIX PTX instructions

**API Design Principles:**
- **Semantic Clarity**: Function names clearly indicate operation level and synchronization model
- **Performance Optimization**: Each tier targets specific hardware acceleration features
- **Type Safety**: Template parameters ensure compile-time correctness
- **Memory Space Awareness**: Clear distinction between pointer types and address spaces

---

**Thread-Level Synchronous Copy:**
Use clean template-based API with all parameters as template arguments:

```cpp
// FROM:
(reinterpret_cast<float4 *>(&dst)[0]) = (reinterpret_cast<float4 *>(&src)[0]);
(reinterpret_cast<float2 *>(&dst)[0]) = (reinterpret_cast<float2 *>(&src)[0]);
(reinterpret_cast<half2 *>(&dst)[0]) = (reinterpret_cast<half2 *>(&src)[0]);

// TO:
thread_copy_sync<dtype, 128, CopyTask::G2S>(&src, &dst);
thread_copy_sync<dtype, 64,  CopyTask::S2R>(&src, &dst); 
thread_copy_sync<dtype, 32,  CopyTask::R2G>(&src, &dst);
```

**CopyTask enum values:**
- `CopyTask::G2S` - Global to Shared memory
- `CopyTask::S2G` - Shared to Global memory  
- `CopyTask::S2R` - Shared to Register memory
- `CopyTask::R2S` - Register to Shared memory
- `CopyTask::G2R` - Global to Register memory
- `CopyTask::R2G` - Register to Global memory

**Data Type Selection Rules:**
1. **If src and dst types are the same**: Use that type directly
2. **If src and dst types differ**: Choose the type based on data semantics, not storage format:
   - Prioritize the more fine-grained, semantically correct data type
   - Consider the kernel's semantic purpose and what data is actually being moved
   - Example: `half` shared memory → `uint32_t` registers → use `half` (data semantics)
3. **Use predefined type aliases** when available: `fp16`, `fp32`, `bf16`, etc.
4. **Be consistent**: Use the same type naming convention throughout the kernel

**Example of correct type selection:**
```cpp
// Shared memory arrays
mem_alloc_shared(fp16, s_a, [BM][BK]);  // half-precision data

// Register arrays (uint32_t for MMA instructions)
mem_alloc_register(uint32_t, RA, [4]);

// Copy operation: semantically moving fp16 data, even though stored in uint32_t registers
warp_copy<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(&s_a[...], RA[0], RA[1], RA[2], RA[3]);
//        ^^^^^ Use fp16 (data semantics), not uint32_t (storage format)
```

### 5. Warp-Level Async Copy Operations

**CP.ASYNC Instructions (Global → Shared Memory ONLY):**
CP.ASYNC hardware only supports Global → Shared direction:

```cpp
// FROM:
asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst_ptr), "l"(&src[addr]), "n"(16));

// TO:  
warp_copy_async<fp16, 128, CopyTask::G2S>(&src[addr], smem_addr);
```

The `warp_copy_async` template API:
- `T`: Data type being copied (fp16, fp32, etc.)  
- `BitWidth`: Total bits to copy (128 bits = 16 bytes)
- `CopyTask`: Only `CopyTask::G2S` supported (hardware limitation)
- **Format**: `warp_copy_async<T, BitWidth, CopyTask::G2S>(global_ptr, smem_addr)`
- **PTX Constraints**: `[shared_addr]` uses `"r"` (32-bit), `[global_addr]` uses `"l"` (64-bit)

**For Shared → Global transfers:**
```cpp
// Use thread_copy_sync instead of warp_copy_async
thread_copy_sync<fp16, 128, CopyTask::S2G>(smem_ptr, global_ptr);
```

**Warp-Level Async Copy Control:**
```cpp
// FROM:
asm volatile("cp.async.commit_group;\n" ::);

// TO:
warp_copy_async_commit_group();
```

```cpp
// FROM:
asm volatile("cp.async.wait_group %0;\n" ::"n"(n));

// TO:
warp_copy_async_wait_group<N>();  // Template version for compile-time constants
```

**Shared Memory Pointer Conversion:**
Use existing functions directly, no need for convenience wrappers:

```cpp
// FROM:
uint32_t smem_ptr = __cvta_generic_to_shared(shared_array);

// TO:
PtrInt smem_ptr = generic_to_shared_ptr(shared_array);
```

### 6. Warp-Level Synchronous Copy Operations

**Matrix Fragment Operations (ldmatrix/stmatrix):**
Use clean template-based API with SmemAddr for shared memory access:

```cpp
// FROM: ldmatrix (Shared to Register)
uint32_t ptr = __cvta_generic_to_shared(&src);
asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" 
    : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3) : "r"(ptr));

// TO:
SmemAddr src_addr = generic_to_shared_addr(&src);
warp_copy_sync<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(src_addr, dst0, dst1, dst2, dst3);
```

```cpp
// FROM: ldmatrix transposed
uint32_t ptr = __cvta_generic_to_shared(&src);
asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
    : "=r"(dst0), "=r"(dst1) : "r"(ptr));

// TO:
SmemAddr src_addr = generic_to_shared_addr(&src);
warp_copy_sync<fp16, 64, CopyTask::S2R, Layout::COL_MAJOR>(src_addr, dst0, dst1);
```

```cpp
// FROM: stmatrix (Register to Shared, SM90+)
uint32_t ptr = __cvta_generic_to_shared(&dst);
asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
    ::"r"(ptr), "r"(src0), "r"(src1), "r"(src2), "r"(src3));

// TO:
SmemAddr dst_addr = generic_to_shared_addr(&dst);
warp_copy_sync<fp16, 128, CopyTask::R2S, Layout::ROW_MAJOR>(dst_addr, src0, src1, src2, src3);
```

**Layout enum values:**
- `Layout::ROW_MAJOR` - Row-major (non-transposed)
- `Layout::COL_MAJOR` - Column-major (transposed)



### 7. MMA Operations
Use clean template-based API with all parameters as template arguments:

```cpp
// FROM:
asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
    : "=r"(rd0), "=r"(rd1) 
    : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1), "r"(rc0), "r"(rc1));

// TO:
mma<MmaShape::M16N8K16, MmaLayout::TN, fp16, fp16>(rd0, rd1, ra0, ra1, ra2, ra3, rb0, rb1, rc0, rc1);
```

**MMA enum values:**
- `MmaShape::M16N8K16` - 16x8x16 matrix shape
- `MmaLayout::TN` - Transposed A, Non-transposed B (row.col)
- `MmaLayout::NT` - Non-transposed A, Transposed B (col.row)

### 8. Synchronization
```cpp
// FROM:
__syncthreads();

// TO:
block_sync();
```

### 9. Warp Shuffle Operations
Replace raw shuffle instructions with semantic primitives:

**🚨 CRITICAL: Sub-Warp vs Full-Warp Operations**

Most Flash Attention and similar algorithms use **sub-warp operations** (e.g., within 4-thread groups), NOT full 32-thread warp operations. Always check the original code for `width` parameters!

**Basic Shuffle Operations:**
```cpp
// FROM:
__shfl_sync(0xffffffff, value, src_lane);                    // Full warp (32 threads)
__shfl_sync(0xffffffff, value, src_lane, 4);               // Sub-warp (4 threads)

// TO:
warp_shuffle(value, src_lane);                              // Full warp (32 threads)
warp_shuffle_width(value, src_lane, 4);                     // Sub-warp (4 threads)
```

**MMA Result Redistribution Pattern:**
```cpp
// FROM: Typical MMA post-processing - NOTE THE WIDTH=4 PARAMETER!
RC0[j][0] = RC[i][j][0];
RC0[j][1] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 1, 4);  // width=4!
RC0[j][2] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 2, 4);  // width=4!
RC0[j][3] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 3, 4);  // width=4!
RC1[j][0] = RC[i][j][1];
RC1[j][1] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 1, 4);  // width=4!
RC1[j][2] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 2, 4);  // width=4!
RC1[j][3] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 3, 4);  // width=4!

// TO: Semantic pattern with correct width
warp_shuffle_spread_dual_x4(RC[i][j][0], RC[i][j][1], RC0[j], RC1[j], lane_id, 4);  // width=4
// OR for explicit clarity:
warp_shuffle_spread_dual_x4_width(RC[i][j][0], RC[i][j][1], RC0[j], RC1[j], lane_id, 4);
```

**Warp Reduction Operations - WATCH FOR WIDTH PARAMETERS!**
```cpp
// FROM: Manual butterfly reduction with XOR - FULL WARP (32 threads)
for (int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset);  // No width = full warp
}

// FROM: Manual butterfly reduction with XOR - SUB-WARP (e.g., 4 threads)  
for (int mask = 4 >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, 4);  // width=4!
}

// TO: Correct semantic reduction based on original code
// Full warp (32 threads):
fp32 result = warp_shuffle_sum(value);                  // Full warp
fp16 result = warp_shuffle_sum(value);                  // Full warp

// Sub-warp (4 threads) - COMMON IN FLASH ATTENTION:  
fp32 result = warp_shuffle_sum_width(value, 4);         // 4-thread sub-warp
fp16 result = warp_shuffle_sum_width(value, 4);         // 4-thread sub-warp
fp32 result = warp_shuffle_max_width(value, 4);         // 4-thread sub-warp

// Type conversion reductions:
fp32 result = warp_shuffle_sum_f16_to_f32(value);       // FP16→FP32 precision  
fp32 result = warp_shuffle_sum_bf16_to_f32(value);      // BF16→FP32 precision
IndexInt result = warp_shuffle_sum_i8_to_i32(value);   // INT8→INT32
```

**🚨 RECOGNITION PATTERNS:**
1. **Check original PTX for width parameter**: `__shfl_xor_sync(mask, value, offset, WIDTH)` 
2. **Look at loop bounds**: `for (int mask = 4 >> 1; ...)` indicates 4-thread sub-warp
3. **Flash Attention typically uses 4-thread sub-warps**
4. **GEMM typically uses full 32-thread warps**

**🔧 FLASH ATTENTION SPECIFIC FIXES:**

The original kernel_expanded.cu vs kernel_prim.cu bug was caused by missing width parameters. Here's how to fix it:

```cpp
// ❌ WRONG - Original buggy DSL conversion
lane_row_max_new[0][0] = warp_shuffle_max<fp32>(lane_row_max_new[0][0], 4);  // 4 is mask, not width!

// ✅ CORRECT - Fixed DSL conversion  
lane_row_max_new[0][0] = warp_shuffle_max_width<fp32>(lane_row_max_new[0][0], 4);  // 4 is width!
lane_row_sum_new[0][0] = warp_shuffle_sum_width<fp32>(lane_row_sum_new[0][0], 4);  // 4 is width!

// Same for MMA result redistribution:
// ❌ WRONG - Missing width parameter
warp_shuffle_spread_dual_x4(R_D[0][j][0], R_D[0][j][1], R_Q[0][0], R_Q[1][0], lane_id);  

// ✅ CORRECT - With width parameter
warp_shuffle_spread_dual_x4(R_D[0][j][0], R_D[0][j][1], R_Q[0][0], R_Q[1][0], lane_id, 4);
```

**Advanced Data Type Support:**
```cpp
// FP8 reductions
fp16 result = warp_shuffle_sum_fp8_e4m3_to_f16(value);  // E4M3→FP16
fp16 result = warp_shuffle_sum_fp8_e5m2_to_f16(value);  // E5M2→FP16

// Max/Min reductions
fp32 max_val = warp_shuffle_max(value);    // Generic max
fp32 max_val = warp_shuffle_max<fp32>(value);  // FP32 (uses fmaxf)
fp32 min_val = warp_shuffle_min(value);    // Generic min
fp32 min_val = warp_shuffle_min<fp32>(value);  // FP32 (uses fminf)
```

**Struct Shuffle Pattern (e.g., Softmax {m, d}):**
```cpp
// FROM: Manual struct shuffle
other.m = __shfl_xor_sync(mask, value.m, stride);
other.d = __shfl_xor_sync(mask, value.d, stride);

// TO: Semantic struct shuffle
using MaxSum = pair_struct<fp32, fp32>;  // {max, sum} for softmax
MaxSum other = warp_shuffle_struct_xor(value, stride);
MaxSum result = warp_shuffle_reduce_struct(value, [](MaxSum a, MaxSum b) {
    return {fmaxf(a.first, b.first), a.second + b.second};
});
```

**Block-Level Reduction Operations:**
```cpp
// FROM: Complex block reduction with shared memory
__shared__ float reduce_smem[NUM_WARPS];
// ... manual warp reduction + shared memory sync ...

// TO: Semantic block reduction
fp32 block_sum = block_reduce_sum<fp32, 256>(thread_value);
fp32 block_max = block_reduce_max<fp32, 256>(thread_value);

// Type conversion reductions
fp32 precise_sum = block_reduce_sum_f16_to_f32<256>(fp16_value);
fp32 precise_sum = block_reduce_sum_bf16_to_f32<256>(bf16_value);
```

**Other Common Patterns:**
```cpp
// Broadcast
fp32 broadcast_val = warp_shuffle_broadcast(value, 0);

// Prefix sum  
fp32 prefix_sum = warp_shuffle_prefix_sum(value);

// Rotation
fp32 rotated = warp_shuffle_rotate(value, 4);
```

### 10. Control Flow - Loop Operations
Replace ALL for loops with DSL loop primitives for consistency:

**Unrolled Loops:**
```cpp
// FROM:
#pragma unroll
for (int i = start; i < end; i += step) {
    // loop body
}

// TO:
serial_range_for(i, start, end, step) {
    // loop body
}
```

**Regular Loops:**
```cpp
// FROM:
for (IndexInt k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    // loop body
}

// TO:
serial_range_for(k, (K_STAGE - 1), NUM_K_TILES, 1) {
    // loop body
}
```

**Complex Start Values:**
```cpp
// FROM:
for (int i = blockIdx.x * blockDim.x; i < N; i += gridDim.x * blockDim.x) {
    // grid-stride loop body
}

// TO:
IndexInt grid_stride_start = blockIdx.x * blockDim.x;
IndexInt grid_stride_step = gridDim.x * blockDim.x;
serial_range_for(i, grid_stride_start, N, grid_stride_step) {
    // grid-stride loop body
}
```

**Key Points:**
- Convert ALL for loops, not just those with `#pragma unroll`
- The `serial_range_for` macro automatically adds `#pragma unroll`
- Support dynamic start/end/step values (runtime computed)
- Use `IndexInt` for loop variables consistently

### 11. Architecture Constants Mapping
**MANDATORY**: Remove ALL local architecture constant definitions and use predefined DSL macros:

```cpp
// FROM: DELETE these patterns
#define WARP_SIZE 32
const int WARP_SIZE = 32;
constexpr int WARP_SIZE = 32;
const int warp_size = 32;
const unsigned int warp_size = 32;
constexpr int MAX_THREADS = 1024;

// TO: Simply use predefined macros (no local definitions needed)
IndexInt warp_id = tid / WARP_SIZE;           // Use predefined WARP_SIZE
IndexInt lane_id = tid % WARP_SIZE;           
constexpr int NUM_THREADS = (TILE_M * TILE_N * WARP_SIZE);
```

**Predefined Architecture Constants** (available in dsl_template.cuh):
- `WARP_SIZE` = 32
- `MAX_THREADS_PER_BLOCK` = 1024  
- `MAX_SHARED_MEM_PER_BLOCK` = 49152 (48KB in bytes)
- `CACHE_LINE_SIZE` = 128 (bytes)
- `MEMORY_BUS_WIDTH` = 128 (bits)
- `GLOBAL_MEM_ALIGN_BYTES` = 128 (128-byte alignment for optimal access)
- `SHARED_MEM_BANK_SIZE` = 32 (number of banks)
- `SHARED_MEM_BANK_WIDTH_BYTES` = 4 (4 bytes per bank)
- `VECTORIZED_ACCESS_WIDTH_BYTES` = 16 (128-bit vectorized access)

**Note**: MMA register counts are **NOT** predefined because they depend on:
- Data types (half, float, etc.)
- Register storage format (uint32_t fragments vs. native types)
- Specific MMA instruction variant
- Implementation choices

Define MMA register constants in your specific kernel as needed.

**Critical Rules**:
1. **DELETE**: Remove any local definition of architecture constants
2. **REPLACE**: Use predefined DSL macros without redefinition  
3. **INCLUDE**: Add `#include "dsl_template.cuh"` at the top
4. **BYTES**: All memory constants are in bytes (not KB)

### 12. Architecture-Aware Development

The DSL automatically detects GPU architecture and enables appropriate features:

**Architecture Detection Macros:**
```cpp
#if __CUDA_ARCH__ >= 700   // SM70+ Volta, Turing, Ampere, Ada, Hopper
    // thread_copy_sync always available
#endif

#if __CUDA_ARCH__ >= 800   // SM80+ Ampere, Ada, Hopper  
    // warp_copy_async (CP.ASYNC) available
    // warp_copy_sync (LDMATRIX S2R) available
#endif

#if __CUDA_ARCH__ >= 900   // SM90+ Hopper
    // warp_copy_sync (STMATRIX R2S) available  
#endif
```

**Conditional Compilation Patterns:**
```cpp
// Optimal code with architecture fallbacks
#if __CUDA_ARCH__ >= 800
    // Use high-performance CP.ASYNC for bulk transfers
    warp_copy_async<fp16, 128, CopyTask::G2S>(&global[addr], smem_addr);
    warp_copy_async_commit_group();
    warp_copy_async_wait_group<0>();
#else
    // Fallback to thread-level synchronous copy
    thread_copy_sync<fp16, 128, CopyTask::G2S>(&global[addr], &shared[idx]);
#endif

// Matrix fragment loading with architecture detection
#if __CUDA_ARCH__ >= 800
    // Use LDMATRIX for optimal tensor core loading
    warp_copy_sync<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(smem_addr, reg0, reg1, reg2, reg3);
#else
    // Fallback to thread-level copies  
    thread_copy_sync<fp16, 32, CopyTask::S2R>(&shared[idx], &reg0);
    thread_copy_sync<fp16, 32, CopyTask::S2R>(&shared[idx+1], &reg1);
    thread_copy_sync<fp16, 32, CopyTask::S2R>(&shared[idx+2], &reg2);
    thread_copy_sync<fp16, 32, CopyTask::S2R>(&shared[idx+3], &reg3);
#endif
```

**Default Architecture Target:**
- **Primary**: SM80+ (Ampere A100, Ada RTX40xx)
- **Secondary**: SM89+ (Ada L40S, RTX40xx Super)
- **Modern**: SM90+ (Hopper H100)

### 13. Utility Functions
```cpp
// FROM:
__device__ __host__ inline int div_ceil(int a, int b) { return (a + b - 1) / b; }
// Usage: div_ceil(a, b)

// TO:
// Usage: ceil_div(a, b)  // macro provided in template
```

### 14. Preserve Original Logic Structure
- Keep the same kernel launch configuration
- Maintain identical thread indexing logic
- Preserve all boundary checking and conditional logic
- Keep the same algorithmic flow and data dependencies


## Example Conversions

### Dynamic Shared Memory Example

**Before:**
```cpp
template <const int TILE_M = 128, const int TILE_N = 128, const int TILE_K = 16>
__global__ void gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
  extern __shared__ half smem[];  // Dynamic shared memory
  
  // Use smem with different interpretations
  half* s_a = smem;                              // Matrix A tile
  half* s_b = smem + TILE_M * TILE_K;           // Matrix B tile  
  uint32_t smem_addr = __cvta_generic_to_shared(smem);  // For PTX instructions
  
  // kernel logic...
}

// Launch with dynamic shared memory size
// gemm_kernel<<<grid, block, shared_mem_size>>>(A, B, C, M, N, K);
```

**After:**
```cpp
template <const TunableInt TILE_M = 128, const TunableInt TILE_N = 128, const TunableInt TILE_K = 16>
__global__ void gemm_kernel(GlobalPtr<fp32> A, GlobalPtr<fp32> B, GlobalPtr<fp32> C, 
                           ShapeInt M, ShapeInt N, ShapeInt K) {
  mem_alloc_shared_dynamic(fp16, smem);  // Dynamic shared memory declaration
  
  // Use typed shared memory pointers - semantic clarity
  SharedPtr<fp16> s_a = shared_ptr_cast<fp16>(smem);                    // Matrix A tile
  SharedPtr<fp16> s_b = shared_ptr_cast<fp16>(smem + TILE_M * TILE_K); // Matrix B tile
  SmemAddr smem_addr = generic_to_shared_addr(smem);                   // For async copy instructions ONLY
  
  // kernel logic...
}

// Launch configuration remains the same  
// gemm_kernel<<<grid, block, shared_mem_size>>>(A, B, C, M, N, K);
```

### Complete Kernel Example

**Before:**
```cpp
template <const int MMA_M = 16, const int TILE_SIZE = 64>
__global__ void kernel(float* A, int M, int N) {
  #define WARP_SIZE 32  // Don't do this!
  const int BM = MMA_M * TILE_SIZE;
  __shared__ half s_a[BM][64];
  uint32_t RC[2] = {0, 0};
  
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  uint32_t smem_ptr = __cvta_generic_to_shared(s_a);
  half* s_ptr = s_a;  // Regular shared memory pointer
  
  for (int i = 0; i < N; ++i) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" 
        ::"r"(dst_ptr), "l"(&A[addr]), "n"(16));
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    __syncthreads();
  }
}
```

**After:**
```cpp
template <const TunableInt MMA_M = 16, const TunableInt TILE_SIZE = 64>
__global__ void kernel(GlobalPtr<fp32> A, ShapeInt M, ShapeInt N) {
  // No local WARP_SIZE definition - use predefined one
  constexpr TileInt BM = MMA_M * TILE_SIZE;
  mem_alloc_shared(fp16, s_a, [BM][64]);
  mem_alloc_register(uint32_t, RC, [2]);
  mem_fill(RC, 0u);
  
  IndexInt tid = threadIdx.x;
  IndexInt warp_id = tid / WARP_SIZE;  // Use predefined WARP_SIZE
  SmemAddr smem_addr = generic_to_shared_addr(s_a);     // For async copy instructions
  SharedPtr<fp16> s_ptr = s_a;                          // Typed shared memory access
  
  serial_range_for(i, 0, N, 1) {
    warp_copy_async<fp16, 128, CopyTask::G2S>(&A[addr], dst_ptr);
    warp_copy_async_commit_group();
    warp_copy_async_wait_group<0>();
    block_sync();
  }
}
```

## Quality Requirements

1. **Functional Equivalence**: The converted kernel must produce identical results
2. **Performance Preservation**: No performance regression from the conversion
3. **Code Clarity**: Use meaningful primitive calls that express intent clearly
4. **Maintainability**: Make the code more readable and easier to modify
5. **Extensibility**: Use patterns that can easily extend to other architectures

## Validation Steps

After conversion, verify:
1. All low-level PTX instructions are replaced with appropriate primitives
2. All memory allocations use the DSL allocation primitives
3. All type declarations use the provided type aliases
4. The kernel maintains the same computational pattern
5. No direct use of reinterpret_cast or inline assembly (except within primitives)
6. All includes and bindings are properly maintained

## Common Patterns to Watch For

1. **Matrix Fragment Loading**: Always becomes `warp_copy_sync(smem_addr, regs...)` for ldmatrix/stmatrix operations
2. **Async Global→Shared Transfers**: Use `warp_copy_async(global_ptr, smem_addr)` for cp.async operations with commit/wait patterns
3. **Accumulator Initialization**: Use `mem_fill` for zero initialization  
4. **Shared Memory Banking**: Preserve original indexing patterns
5. **Thread Mapping**: Keep original thread-to-data mapping logic intact
6. **Boundary Checks**: Maintain all safety checks and early returns
7. **Loop Operations**: ALL for loops must become `serial_range_for` (not just unrolled ones)
8. **Complex Loop Bounds**: Pre-compute dynamic start/end values when needed
9. **Pointer Type Selection**: Use SmemAddr for PTX instruction addresses (cp.async, ldmatrix), SharedPtr/GlobalPtr for C++ operations
10. **Warp Shuffle Patterns**: Replace raw __shfl_sync with semantic primitives (broadcast, reduce, redistribute, etc.)
   - **🚨 CRITICAL**: Always check for width parameters in original shuffle calls!
   - **Flash Attention**: Typically uses 4-thread sub-warps → use `_width` variants
   - **GEMM**: Typically uses full 32-thread warps → use standard variants
11. **MMA Result Processing**: Use `warp_shuffle_spread_x4` or `warp_shuffle_spread_dual_x4` for typical redistribution patterns
   - **With width param**: `warp_shuffle_spread_dual_x4(src0, src1, dst0, dst1, lane_id, 4)`  
   - **Full warp**: `warp_shuffle_spread_dual_x4(src0, src1, dst0, dst1, lane_id)`
12. **Multi-Type Reductions**: Use type-specific reductions with correct width
   - **Sub-warp**: `warp_shuffle_sum_width<fp32>(value, 4)`, `warp_shuffle_max_width<fp32>(value, 4)`
   - **Full warp**: `warp_shuffle_sum<fp16>(value)`, `warp_shuffle_sum_f16_to_f32(value)`
13. **Block-Level Reductions**: Replace manual shared memory patterns with `block_reduce_sum<T, NUM_THREADS>`
14. **Struct Data Shuffle**: Use `pair_struct<T1,T2>` and `warp_shuffle_struct_xor` for multi-field structs
15. **Copy Operation Hierarchy**: 
   - **Thread-level**: `thread_copy_sync(src_ptr, dst_ptr)` - all directions, all architectures
   - **Warp-level async**: `warp_copy_async(global_ptr, smem_addr)` - G2S only, SM80+
   - **Warp-level sync**: `warp_copy_sync(smem_addr, regs...)` - S2R (SM80+), R2S (SM90+)

16. **Architecture-Specific Capabilities**:
   - **SM70-79 (Volta/Turing)**: Only `thread_copy_sync` available
   - **SM80-89 (Ampere/Ada)**: Add `warp_copy_async` (CP.ASYNC), `warp_copy_sync` (LDMATRIX)
   - **SM90+ (Hopper)**: Add `warp_copy_sync` (STMATRIX) support

### Specific Loop Conversion Examples

**Non-zero Start Values:**
```cpp
// ❌ WRONG - Direct for loop usage
for (IndexInt k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    // pipeline loop body
}

// ✅ CORRECT - DSL loop primitive
serial_range_for(k, (K_STAGE - 1), NUM_K_TILES, 1) {
    // pipeline loop body  
}
```

**Grid-Stride Loops:**
```cpp  
// ❌ WRONG - Complex expression in for loop
for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total_elements; tid += gridDim.x * blockDim.x) {
    // grid-stride loop body
}

// ✅ CORRECT - Pre-compute expressions  
IndexInt tid_start = blockIdx.x * blockDim.x + threadIdx.x;
IndexInt tid_step = gridDim.x * blockDim.x;
serial_range_for(tid, tid_start, total_elements, tid_step) {
    // grid-stride loop body
}
```

### Specific Pointer Type Selection Examples

**SmemAddr for Both Async and Sync PTX Instructions:**
```cpp
// ✅ CORRECT - SmemAddr for async copy instructions  
SmemAddr load_addr = generic_to_shared_addr(&s_a[stage][m][k]);
warp_copy_async<fp16, 128, CopyTask::G2S>(&A[addr], load_addr);
```

**SmemAddr for Matrix Fragment Operations:**
```cpp  
// ✅ CORRECT - SmemAddr for warp sync copy operations
SmemAddr lane_addr = generic_to_shared_addr(&s_a[m][k]);
warp_copy_sync<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(lane_addr, reg0, reg1, reg2, reg3);
```

**Simple Recognition Pattern:**
- **For PTX instructions (cp.async, ldmatrix, stmatrix)?** → Use `SmemAddr`
- **For regular C++ operations (indexing, arithmetic)?** → Use `SharedPtr<T>` / `GlobalPtr<T>`
- **At call site conversion:** `SmemAddr addr = generic_to_shared_addr(shared_ptr)`

### Specific Warp Shuffle Pattern Examples

**MMA Result Redistribution (the most common pattern):**
```cpp
// STEP 1: Check original code for width parameters
// ❌ WRONG ANALYSIS - Missing width parameters
RC0[j][1] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 1);
RC0[j][2] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 2);

// ✅ CORRECT ANALYSIS - Original code has width=4
RC0[j][1] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 1, 4);  // width=4!
RC0[j][2] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 2, 4);  // width=4!

// STEP 2: Apply correct DSL conversion
// ❌ WRONG - Missing width parameter
serial_range_for(j, 0, WARP_TILE_N, 1) {
  warp_shuffle_spread_dual_x4(RC[i][j][0], RC[i][j][1], RC0[j], RC1[j], lane_id);  // MISSING WIDTH!
}

// ✅ CORRECT - With width parameter for sub-warp operation
serial_range_for(j, 0, WARP_TILE_N, 1) {
  warp_shuffle_spread_dual_x4(RC[i][j][0], RC[i][j][1], RC0[j], RC1[j], lane_id, 4);  // width=4
}

// Alternative explicit version:
serial_range_for(j, 0, WARP_TILE_N, 1) {
  warp_shuffle_spread_dual_x4_width(RC[i][j][0], RC[i][j][1], RC0[j], RC1[j], lane_id, 4);
}
```

**Warp Reduction Pattern:**
```cpp
// STEP 1: Analyze original reduction loop bounds
// ❌ WRONG ANALYSIS - Full warp reduction
for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(0xffffffff, sum, offset);  // No width = 32 threads
}

// ✅ CORRECT ANALYSIS - Sub-warp reduction (common in Flash Attention)
for (int mask = 4 >> 1; mask >= 1; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask, 4);  // width=4!
}

// STEP 2: Apply correct DSL conversion
// ❌ WRONG - Wrong reduction scope (32 threads instead of 4)
fp32 total = warp_shuffle_sum(sum);  // This does 32-thread reduction!

// ✅ CORRECT - Sub-warp reduction (4 threads)
fp32 total = warp_shuffle_sum_width(sum, 4);  // This does 4-thread reduction

// For type-specific optimizations:
fp32 max_val = warp_shuffle_max_width<fp32>(value, 4);     // 4-thread max
fp16 sum_f16 = warp_shuffle_sum_width<fp16>(value, 4);     // 4-thread sum
```

**Broadcast Pattern:**
```cpp
// ❌ WRONG - Direct shuffle call
fp32 shared_value = __shfl_sync(0xffffffff, value, 0);

// ✅ CORRECT - Semantic broadcast
fp32 shared_value = warp_shuffle_broadcast(value, 0);
```

**Multi-Data-Type Reduction Examples:**
```cpp
// ❌ WRONG - Manual FP16 reduction
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// ✅ CORRECT - Type-aware semantic reduction
fp16 sum_f16 = warp_shuffle_sum(fp16_value);           // Pure FP16
fp32 sum_f32 = warp_shuffle_sum_f16_to_f32(fp16_value); // FP16→FP32 precision
```

**Block Reduction Pattern:**
```cpp
// ❌ WRONG - Manual block reduction with shared memory
constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
__shared__ float reduce_smem[NUM_WARPS];
// ... complex warp reduction + shared memory sync ...

// ✅ CORRECT - Semantic block reduction
fp32 final_sum = block_reduce_sum<fp32, 256>(thread_value);

// Different data types
fp32 precise_sum = block_reduce_sum_f16_to_f32<256>(fp16_value);  // FP16 input, FP32 accumulation
fp32 precise_sum = block_reduce_sum_bf16_to_f32<256>(bf16_value); // BF16 input, FP32 accumulation
```

**Complex Data Structure Shuffle (Softmax-style):**
```cpp
// ❌ WRONG - Manual struct member shuffle
struct MaxSum {
  float m, d;  
};
other.m = __shfl_xor_sync(mask, value.m, stride);
other.d = __shfl_xor_sync(mask, value.d, stride);

// ✅ CORRECT - Semantic struct shuffle
using MaxSum = pair_struct<fp32, fp32>;
MaxSum other = warp_shuffle_struct_xor(value, stride);
MaxSum final = warp_shuffle_reduce_struct(value, [](MaxSum a, MaxSum b) {
    return {fmaxf(a.first, b.first), a.second + b.second};
});
```

## Best Practices for Memory Pointer Types

### Simplified Type Selection Guide

**Single Decision Point:**

1. **Will this address be used in PTX instruction requiring shared memory address?**
   - YES → Use `SmemAddr` (direct __cvta_generic_to_shared result or computed from it)
   - NO → Use appropriate typed pointer (`SharedPtr<T>`, `GlobalPtr<T>`, `RegisterPtr<T>`)

### Memory Space Selection:
- **Global memory** → Use `GlobalPtr<T>` 
- **Shared memory** → Use `SharedPtr<T>`
- **Register arrays** → Use `RegisterPtr<T>`
- **Dynamic shared memory** → Use `shared_ptr_cast<dtype>(raw_smem)`

### Code Clarity Benefits

**Before (Confusing):**
```cpp
uint32_t ptr1 = array1;                    // What kind of pointer?
uint32_t ptr2 = __cvta_generic_to_shared(&s_a[m][k]);  // Address conversion
half* ptr3 = dynamic_smem;                 // Dynamic shared memory
```

**After (Clear Intent):**
```cpp
SharedPtr<fp16> ptr1 = array1;             // Clearly typed pointer for calculations
SmemAddr ptr2 = generic_to_shared_addr(&s_a[m][k]); // Clearly for copy instruction
SharedPtr<fp16> ptr3 = shared_ptr_cast<fp16>(dynamic_smem); // Clearly typed shared memory
```

### Benefits of This Approach:
1. **Ultra-simple rule** - Only one decision point
2. **Type safety** - Always know what data type the pointer points to
3. **Clear purpose** - SmemAddr is only for copy instructions, everything else uses typed pointers
4. **No complex decisions** - No need to distinguish byte vs element offsets

## 🎆 API DESIGN IMPROVEMENTS (v3.0)

The DSL has been significantly enhanced with a cleaner, more semantic API design that eliminates previous inconsistencies:

### 🚀 Major Improvements:

1. **Hierarchical Copy Operations**: 
   - **Enhancement**: Clear separation between thread-level, warp-async, and warp-sync operations
   - **Benefits**: Semantic clarity, performance optimization targeting, better code organization
   - **New API**: `thread_copy_sync`, `warp_copy_async`, `warp_copy_sync` with distinct use cases

2. **Eliminated API Ambiguity**:
   - **Previous Issue**: Confusing mix of `thread_copy_async` and `warp_copy` nomenclature
   - **Solution**: Consistent naming that reflects operation level and synchronization model
   - **Result**: Code that clearly expresses programmer intent and hardware utilization

3. **Memory Address Type Safety**:
   - **Enhancement**: Refined SmemAddr usage to only async copy operations
   - **Benefits**: Clearer separation of concerns, reduced pointer type confusion
   - **Rule**: SmemAddr for cp.async only, SharedPtr/GlobalPtr for everything else

4. **Fixed Shuffle Operation Bugs**:
   - **Bug**: `warp_shuffle_spread_x4()` and `warp_shuffle_spread_dual_x4()` lacked width parameters
   - **Bug**: All `warp_shuffle_*` reductions operated on 32 threads instead of sub-warps
   - **Bug**: Parameter semantic confusion between mask and width
   - **Fix**: Complete API with explicit width variants and correct default behaviors

### 🔧 Migration Guide:

Key API changes for the new hierarchical design:
```cpp
// OLD API:
thread_copy_async<fp16, 128, CopyTask::G2S>(&src, dst_ptr);            // Confusing level
warp_copy<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(&src, ...);     // Ambiguous sync/async
warp_shuffle_spread_dual_x4(src0, src1, dst0, dst1, lane_id);          // Missing width

// NEW API:
warp_copy_async<fp16, 128, CopyTask::G2S>(&src, dst_ptr);              // Clear async warp-level
warp_copy_sync<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(&src, ..); // Clear sync warp-level
warp_shuffle_spread_dual_x4(src0, src1, dst0, dst1, lane_id, 4);       // Explicit width=4
```

### 📋 Validation Checklist:

For every converted kernel, verify:
- [ ] Memory operations use appropriate hierarchy level (thread/warp-async/warp-sync)
- [ ] All shuffle operations have correct width parameters
- [ ] Sub-warp reductions use `_width` variants  
- [ ] Flash Attention uses 4-thread sub-warps
- [ ] GEMM uses full 32-thread warps
- [ ] SmemAddr only used for async copy operations
- [ ] Consistent use of semantic pointer types throughout kernel

---

## 💡 New API Design Summary

### Memory Copy Operations - Three Clear Tiers:

1. **`thread_copy_sync<T, BitWidth, CopyTask>(src, dst)`**
   - **When**: Simple memory transfers within single thread
   - **Examples**: Basic data movement, element copies
   - **All memory spaces supported**

2. **`warp_copy_async<T, BitWidth, CopyTask::G2S>(global_ptr, smem_addr)`**
   - **When**: High-throughput Global → Shared pipeline (cp.async)
   - **Examples**: Loading matrix tiles from global to shared memory
   - **Requires**: `warp_copy_async_commit_group()` and `warp_copy_async_wait_group<N>()`
   - **Address Type**: Global pointer + `SmemAddr` for destination
   - **Limitation**: Only G2S direction supported by CP.ASYNC hardware

3. **`warp_copy_sync<T, BitWidth, CopyTask, Layout>(smem_addr, regs...)`**
   - **When**: Matrix fragment operations (ldmatrix/stmatrix)
   - **Examples**: Loading matrix tiles for MMA, tensor core operations
   - **Address Type**: Use `SmemAddr` for shared memory access
   - **Formats**: `warp_copy_sync<T, BitWidth, CopyTask::S2R, Layout>(smem_addr, regs...)`

### Key Decision Points:

🤔 **"What kind of memory operation am I doing?"**
- **Simple copy (any direction)** → `thread_copy_sync`
- **Bulk async Global → Shared** → `warp_copy_async` 
- **Matrix fragment (ldmatrix/stmatrix)** → `warp_copy_sync`

🤔 **"What address type should I use?"**
- **PTX instructions (cp.async, ldmatrix, stmatrix)** → `SmemAddr` 
- **Regular C++ operations (indexing, arithmetic)** → `SharedPtr<T>` / `GlobalPtr<T>`

🤔 **"Warp shuffle width parameter?"**
- **Flash Attention** → Usually 4-thread sub-warps
- **GEMM** → Usually full 32-thread warps
- **Check original** → Look for width parameter in `__shfl_*` calls

### Benefits of New Design:
- **💯 Performance**: Each operation targets optimal hardware path
- **📈 Clarity**: Function names clearly express intent and synchronization
- **🔒 Safety**: Type system prevents common address/pointer mistakes
- **🚀 Maintainability**: Consistent API reduces cognitive load

---

Now convert the provided CUDA kernel following these rules exactly: