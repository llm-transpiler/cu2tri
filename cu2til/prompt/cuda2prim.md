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
SmemAddr bad_addr = some_array;  // NO! SmemAddr is for copy addresses only

// ✅ CORRECT - Use semantic pointer types for memory access  
SharedPtr<fp16> good_shared_ptr = shared_array;       // Shared memory access
GlobalPtr<fp32> good_global_ptr = global_array;       // Global memory access
SmemAddr copy_addr = generic_to_shared_addr(shared_array); // Copy instruction address ONLY
```

**Memory Access Pattern:**
1. **Use `SharedPtr<T>` / `GlobalPtr<T>`** → ALL memory operations, array indexing, pointer arithmetic
2. **Use `SmemAddr`** → ONLY shared memory addresses for PTX instructions (ldmatrix, cp.async) - direct __cvta_generic_to_shared results or computed from them
3. **Never mix** → Don't use SmemAddr for normal C++ pointer operations

**Simple Decision Rule:**
**Will this address be used in PTX instruction requiring shared memory address?** → Use `SmemAddr`. **Everything else?** → Use `SharedPtr<T>` / `GlobalPtr<T>`

**PTX Instruction Address Examples:**
```cpp
// ✅ CORRECT - Direct __cvta_generic_to_shared result
SmemAddr smem_base_addr = generic_to_shared_addr(s_a);

// ✅ CORRECT - Computed address for PTX instruction  
SmemAddr load_addr = (smem_base_addr + 
    (k * stage_offset + m * (BK + A_PAD) + k_offset) * sizeof(fp16));
thread_copy_async<fp16, 128, CopyTask::G2S>(&A[addr], load_addr);

// ✅ CORRECT - SharedPtr for all other operations
SharedPtr<fp16> lane_ptr = s_a + (stage * stage_offset + m * stride + k);
warp_copy<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(lane_ptr, reg0, reg1, reg2, reg3);
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

**Thread-Level Copy:**
Use clean template-based API with all parameters as template arguments:

```cpp
// FROM:
(reinterpret_cast<float4 *>(&dst)[0]) = (reinterpret_cast<float4 *>(&src)[0]);
(reinterpret_cast<float2 *>(&dst)[0]) = (reinterpret_cast<float2 *>(&src)[0]);
(reinterpret_cast<half2 *>(&dst)[0]) = (reinterpret_cast<half2 *>(&src)[0]);

// TO:
thread_copy<dtype, 128, CopyTask::G2S>(&src, &dst);
thread_copy<dtype, 64,  CopyTask::S2R>(&src, &dst); 
thread_copy<dtype, 32,  CopyTask::R2G>(&src, &dst);
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

**Async Copy Operations:**
Use clean template-based API with all parameters as template arguments:

```cpp
// FROM:
asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst_ptr), "l"(&src[addr]), "n"(16));

// TO:
thread_copy_async<fp16, 128, CopyTask::G2S>(&src[addr], dst_ptr);
```

The `thread_copy_async` template API:
- `T`: Data type being copied (fp16, fp32, etc.)  
- `BitWidth`: Total bits to copy (128 bits = 16 bytes)
- `CopyTask`: Copy task enum (CopyTask::G2S, CopyTask::S2G, CopyTask::G2R, CopyTask::R2G, CopyTask::S2R, CopyTask::R2S)
- Arguments: source pointer, destination pointer

**Async Copy Control:**
```cpp
// FROM:
asm volatile("cp.async.commit_group;\n" ::);

// TO:
thread_copy_async_commit_group();
```

```cpp
// FROM:
asm volatile("cp.async.wait_group %0;\n" ::"n"(n));

// TO:
thread_copy_async_wait_group<N>();  // Template version for compile-time constants
```

**Shared Memory Pointer Conversion:**
Use existing functions directly, no need for convenience wrappers:

```cpp
// FROM:
uint32_t smem_ptr = __cvta_generic_to_shared(shared_array);

// TO:
PtrInt smem_ptr = generic_to_shared_ptr(shared_array);
```

**Warp-Level Copy (ldmatrix/stmatrix operations):**
Use clean template-based API with CopyTask and layout as template parameters:

```cpp
// FROM: ldmatrix (Shared to Register)
uint32_t ptr = __cvta_generic_to_shared(&src);
asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" 
    : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3) : "r"(ptr));

// TO:
warp_copy<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(&src, dst0, dst1, dst2, dst3);
```

```cpp
// FROM: ldmatrix transposed
uint32_t ptr = __cvta_generic_to_shared(&src);
asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
    : "=r"(dst0), "=r"(dst1) : "r"(ptr));

// TO:
warp_copy<fp16, 64, CopyTask::S2R, Layout::COL_MAJOR>(&src, dst0, dst1);
```

```cpp
// FROM: stmatrix (Register to Shared, SM90+)
uint32_t ptr = __cvta_generic_to_shared(&dst);
asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
    ::"r"(ptr), "r"(src0), "r"(src1), "r"(src2), "r"(src3));

// TO:
warp_copy<fp16, 128, CopyTask::R2S, Layout::ROW_MAJOR>(&dst, src0, src1, src2, src3);
```

**Layout enum values:**
- `Layout::ROW_MAJOR` - Row-major (non-transposed)
- `Layout::COL_MAJOR` - Column-major (transposed)

### 5. Async Copy Operations

**CP.ASYNC Instructions:**
Use clean template-based API with all parameters as template arguments:

```cpp
// FROM:
asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" 
    ::"r"(dst_ptr), "l"(&src[addr]), "n"(16));

// TO:  
thread_copy_async<fp16, 128, CopyTask::G2S>(&src[addr], dst_ptr);
```

**Async Copy Control:**
```cpp
// FROM:
asm volatile("cp.async.commit_group;\n" ::);

// TO:
thread_copy_async_commit_group();
```

```cpp
// FROM:
asm volatile("cp.async.wait_group %0;\n" ::"n"(N));

// TO:
thread_copy_async_wait_group<N>();  // Template version for compile-time constants
```

### 6. MMA Operations
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

### 7. Synchronization
```cpp
// FROM:
__syncthreads();

// TO:
block_sync();
```

### 8. Warp Shuffle Operations
Replace raw shuffle instructions with semantic primitives:

**Basic Shuffle Operations:**
```cpp
// FROM:
__shfl_sync(0xffffffff, value, src_lane);

// TO:
warp_shuffle(value, src_lane);
```

**MMA Result Redistribution Pattern:**
```cpp
// FROM: Typical MMA post-processing
RC0[j][0] = RC[i][j][0];
RC0[j][1] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 1);
RC0[j][2] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 2);
RC0[j][3] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 3);
RC1[j][0] = RC[i][j][1];
RC1[j][1] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 1);
RC1[j][2] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 2);
RC1[j][3] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 3);

// TO: Semantic pattern
warp_shuffle_spread_dual_x4(RC[i][j][0], RC[i][j][1], RC0[j], RC1[j], lane_id);
```

**Warp Reduction Operations (Type-Aware):**
```cpp
// FROM: Manual butterfly reduction with XOR
for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
}

// TO: Type-aware semantic reduction
fp32 result = warp_shuffle_sum(value);  // FP32
fp16 result = warp_shuffle_sum(value);  // FP16 (uses __hadd)
bf16 result = warp_shuffle_sum(value);  // BF16 (uses __hadd)
fp32 result = warp_shuffle_sum_f16_to_f32(value);  // FP16→FP32 precision
fp32 result = warp_shuffle_sum_bf16_to_f32(value); // BF16→FP32 precision
IndexInt result = warp_shuffle_sum_i8_to_i32(value);  // INT8→INT32
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

### 9. Control Flow - Loop Operations
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

### 10. Architecture Constants Mapping
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

### 11. Utility Functions
```cpp
// FROM:
__device__ __host__ inline int div_ceil(int a, int b) { return (a + b - 1) / b; }
// Usage: div_ceil(a, b)

// TO:
// Usage: ceil_div(a, b)  // macro provided in template
```

### 12. Preserve Original Logic Structure
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
  SmemAddr smem_addr = generic_to_shared_addr(smem);                       // For copy instructions ONLY
  
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
  SmemAddr smem_addr = generic_to_shared_addr(s_a);      // For copy instructions
  SharedPtr<fp16> s_ptr = s_a;                       // Typed shared memory access
  
  serial_range_for(i, 0, N, 1) {
    thread_copy_async<fp16, 128, CopyTask::G2S>(&A[addr], dst_ptr);
    thread_copy_async_commit_group();
    thread_copy_async_wait_group<0>();
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

1. **Matrix Fragment Loading**: Always becomes `warp_copy` with appropriate layout
2. **Accumulator Initialization**: Use `mem_fill` for zero initialization  
3. **Shared Memory Banking**: Preserve original indexing patterns
4. **Thread Mapping**: Keep original thread-to-data mapping logic intact
5. **Boundary Checks**: Maintain all safety checks and early returns
6. **Loop Operations**: ALL for loops must become `serial_range_for` (not just unrolled ones)
7. **Complex Loop Bounds**: Pre-compute dynamic start/end values when needed
8. **Pointer Type Selection**: Use SmemAddr only for copy instruction addresses, SharedPtr/GlobalPtr for everything else
9. **Warp Shuffle Patterns**: Replace raw __shfl_sync with semantic primitives (broadcast, reduce, redistribute, etc.)
10. **MMA Result Processing**: Use `warp_shuffle_spread_x4` or `warp_shuffle_spread_dual_x4` for typical redistribution patterns  
11. **Multi-Type Reductions**: Use type-specific reductions (`warp_shuffle_sum<fp16>`, `warp_shuffle_sum_f16_to_f32`, etc.)
12. **Block-Level Reductions**: Replace manual shared memory patterns with `block_reduce_sum<T, NUM_THREADS>`
13. **Struct Data Shuffle**: Use `pair_struct<T1,T2>` and `warp_shuffle_struct_xor` for multi-field structs

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

**Shared Memory Address for Copy Instructions:**
```cpp
// ✅ CORRECT - SmemAddr for copy instructions  
SmemAddr load_addr = generic_to_shared_addr(&s_a[stage][m][k]);
thread_copy_async<fp16, 128, CopyTask::G2S>(&A[addr], load_addr);
```

**All Other Operations Use Typed Pointers:**
```cpp  
// ✅ CORRECT - SharedPtr for all other operations
SharedPtr<fp16> lane_ptr = s_a + (stage * stage_offset + m * stride + k);
warp_copy<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(lane_ptr, reg0, reg1, reg2, reg3);
```

**Simple Recognition Pattern:**
- **For PTX instructions (ldmatrix, cp.async), need shared memory address?** → Use `SmemAddr`
- **Everything else?** → Use `SharedPtr<T>` / `GlobalPtr<T>`

### Specific Warp Shuffle Pattern Examples

**MMA Result Redistribution (the most common pattern):**
```cpp
// ❌ WRONG - Manual shuffle operations
serial_range_for(j, 0, WARP_TILE_N, 1) {
  RC0[j][0] = RC[i][j][0];
  RC1[j][0] = RC[i][j][1];
  RC0[j][1] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 1);
  RC0[j][2] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 2);
  RC0[j][3] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 3);
  RC1[j][1] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 1);
  RC1[j][2] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 2);
  RC1[j][3] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 3);
}

// ✅ CORRECT - Semantic shuffle pattern
serial_range_for(j, 0, WARP_TILE_N, 1) {
  warp_shuffle_spread_dual_x4(RC[i][j][0], RC[i][j][1], RC0[j], RC1[j], lane_id);
}
```

**Warp Reduction Pattern:**
```cpp
// ❌ WRONG - Manual butterfly reduction  
for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    fp32 other = __shfl_down_sync(0xffffffff, sum, offset);
    sum += other;
}

// ✅ CORRECT - Semantic reduction
fp32 total = warp_shuffle_sum(sum);
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

Now convert the provided CUDA kernel following these rules exactly: