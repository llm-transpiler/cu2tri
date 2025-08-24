# Primitive to Coordinate-based DSL Transformation Prompt

## Task

Transform CUDA kernels that use tile-level DSL primitives into coordinate-based DSL versions. The transformation preserves semantic equivalence while replacing direct memory indexing with coordinate-based abstractions for memory access patterns.

## Background

You are given a CUDA kernel that uses tile-level DSL primitives (thread_copy, warp_copy, mma, etc.) with direct memory indexing. Your task is to convert it to use coordinate-based abstractions that provide a higher-level, more maintainable approach to expressing memory access patterns and coordinate system transformations.

## Reference Documentation

**CRITICAL**: Before proceeding, consult the comprehensive [Coordinate DSL API Reference](./coordinate_dsl_api_reference.md) for precise API specifications, syntax, and usage examples. This transformation guide provides the process, while the API reference provides the exact implementation details.

## Input Format
- CUDA kernel file using tile-level DSL primitives from dsl_template.cuh
- Direct memory indexing and address calculations
- Standard memory access patterns (g2s, s2r, r2s, s2g)

## Output Format
- Coordinate-based CUDA kernel including coord_template.cuh header
- All memory indexing replaced with coordinate-based abstractions
- Explicit coordinate tile and coordinate space definitions
- Preserved DSL primitives with coordinate-based addressing

## Conversion Rules

### 1. Header Addition
```cpp
// Add after dsl_template.cuh include:
#include "coord_template.cuh"
```

## Core Transformation Principles

### 1. Semantic Preservation
- **CRITICAL**: The transformed code must be semantically equivalent to the original
- All memory access patterns, computation order, and synchronization points must be preserved
- Performance characteristics should remain comparable or improved

### 2. Abstraction Level
- Replace low-level index arithmetic with coordinate-based abstractions
- Maintain explicit type information throughout the transformation
- Use multi-dimensional coordinate systems to express memory access patterns naturally

### 3. Generality
- The transformation should work for various compute primitives:
  - **Matrix Operations**: GEMM, GEMV, matrix transpose, batched operations
  - **Convolution Operations**: 1D/2D/3D convolutions, depthwise/grouped convolutions
  - **Reduction Operations**: Softmax, layer norm, sum/max/min reductions
  - **Element-wise Operations**: Activation functions, broadcasting operations
  - **Fused Operations**: Any combination of the above

### 2. Memory Copy Pattern Transformation

Memory copy operations are classified by **execution model**, not memory type, as this reflects hardware capabilities:

#### Thread-Level Copy (`thread_copy`)
**Execution Model**: Single thread operation  
**Supported Memory Transfers**: Global ↔ Shared ↔ Register (any combination)
- **G2S/S2G**: Global ↔ Shared Memory
- **G2R/R2G**: Global ↔ Register Memory  
- **S2R/R2S**: Shared ↔ Register Memory

**Usage Pattern**: `thread_copy(dtype, bits, "task", src_ptr, dst_ptr)`

#### Warp-Level Copy (`warp_copy`)  
**Execution Model**: Warp-collective operation (32 threads cooperate)  
**Primary Use Case**: Shared → Register with `ldmatrix` instruction
- **S2R**: Shared Memory → Register (warp-collective `ldmatrix`)

**Usage Pattern**: `warp_copy(dtype, bits, "s2r", layout, src_ptr, dst_regs...)`

**Key Distinction**: Use `thread_copy` for individual thread operations, `warp_copy` for warp-collective operations that require hardware synchronization.

### 3. Coordinate Tile Definition

Define coordinate tiles based on algorithm dimensionality:

**Common Tile Types:**
- **2D Matrix Operations** (GEMM, 2D Conv): `CoordTile2D<M, N>`
- **1D Vector Operations** (GEMV, Reductions): `CoordTile1D<N>`  
- **3D Tensor Operations** (3D Conv): `CoordTile3D<D, H, W>`

**Tile Granularity:**
- **Thread-level**: Individual thread work (`reg_tile<2, 2>()`, `unit_tile()`)
- **Warp-level**: Collective operations (`warp_tile<16, 8>()`)  
- **Block-level**: Shared memory tiles (`tile2d<MMA_M, MMA_K>()`)

## Transformation Process

### Step 1: Identify Memory Copy Operations
Scan the kernel for all memory copy operations and classify them:
- **Thread-level copies**: `thread_copy(...)` with direct memory indexing
- **Warp-level copies**: `warp_copy(...)` with direct shared memory addressing

### Step 2: Apply Coordinate Transformation
For each identified memory copy operation:

1. **Define appropriate coordinate tiles** based on operation granularity
2. **Create coordinate spaces** for source and destination memory
3. **Map thread/warp coordinates** to the coordinate space
4. **Replace direct pointers** with coordinate-based addressing

### Step 3: Detailed Transformation Examples

**Both this guide and the [API Reference](./coordinate_dsl_api_reference.md) provide complete examples. Use both for comprehensive understanding.**

## Memory Access Transformation Patterns

### Pattern 1: Global-to-Shared Memory Copy (G2S)

**Before:**
```cuda
int load_gmem_addr = load_gmem_m * K + load_gmem_k;
thread_copy(half, 128, "g2s", &A[load_gmem_addr], &s_a[load_smem_m][load_smem_k]);
```

**After:**
```cuda
CoordTile2D<MMA_M, MMA_K> g2s_tile = tile2d<MMA_M, MMA_K>();
CoordSpace<CoordTile2D<MMA_M, MMA_K>> g2s_coord_space = make_coord_space(g2s_tile);
CoordSpace<CoordTile2D<MMA_M, MMA_K>> g2s_gmem_space = g2s_coord_space.step(offset(by, k), g2s_tile);
CoordSpace<CoordTile2D<MMA_M, MMA_K>> g2s_smem_space = g2s_coord_space.step(offset(0, 0), g2s_tile);
Coord<CoordSpace<CoordTile2D<MMA_M, MMA_K>>> thread_gmem_coord = thread_coord_map(g2s_gmem_space, coord(tid / 2, (tid % 2) * 8));
Coord<CoordSpace<CoordTile2D<MMA_M, MMA_K>>> thread_smem_coord = thread_coord_map(g2s_smem_space, coord(tid / 2, (tid % 2) * 8));

PtrOffset gmem_offset = make_ptr_offset(thread_gmem_coord, stride(K, 1));
thread_copy(half, 128, "g2s", get_off_ptr(A, gmem_offset), get_crd_ptr(s_a, thread_smem_coord));
```

### Pattern 2: Shared-to-Register Copy (S2R) - Warp-Level

**Before:**
```cuda
warp_copy(uint32_t, 128, "s2r", 'n', &s_a[lane_id % 16][(lane_id / 16) * 8], RA[0], RA[1], RA[2], RA[3]);
```

**After:**
```cuda
CoordTile2D<16, 8> s2r_warp_tile = warp_tile<16, 8>();
CoordSpace<CoordTile2D<16, 8>> s2r_coord_space = make_coord_space(s2r_warp_tile);
CoordSpace<CoordTile2D<16, 8>> s2r_smem_space = s2r_coord_space.step(offset(0, 0), s2r_warp_tile);
Coord<CoordSpace<CoordTile2D<16, 8>>> thread_smem_coord = thread_coord_map(s2r_smem_space, coord(lane_id % 16, (lane_id / 16) * 8));

warp_copy(uint32_t, 128, "s2r", 'n', get_crd_ptr(s_a, thread_smem_coord), RA[0], RA[1], RA[2], RA[3]);
```

### Pattern 3: Register-to-Shared Copy (R2S) - Thread-Level

**Before:**
```cuda
thread_copy(half, 32, "r2s", &RC[0], &s_c[lane_id / 4][(lane_id % 4) * 2]);
thread_copy(half, 32, "r2s", &RC[1], &s_c[lane_id / 4 + 8][(lane_id % 4) * 2]);
```

**After:**
```cuda
CoordTile2D<2, 2> r2s_reg_tile = reg_tile<2, 2>();
CoordTile2D<MMA_M, MMA_N> r2s_smem_tile = tile2d<MMA_M, MMA_N>();
CoordSpace<CoordTile2D<2, 2>> r2s_reg_coord_space = make_coord_space(r2s_reg_tile);
CoordSpace<CoordTile2D<MMA_M, MMA_N>> r2s_smem_coord_space = make_coord_space(r2s_smem_tile);

Coord<CoordSpace<CoordTile2D<2, 2>>> thread_reg_coord_0 = thread_coord_map(r2s_reg_coord_space.step(offset(0, 0), unit_tile()), coord(0, _));
Coord<CoordSpace<CoordTile2D<2, 2>>> thread_reg_coord_1 = thread_coord_map(r2s_reg_coord_space.step(offset(1, 0), unit_tile()), coord(0, _));
Coord<CoordSpace<CoordTile2D<MMA_M, MMA_N>>> thread_smem_coord_0 = thread_coord_map(r2s_smem_coord_space.step(offset(0, 0), unit_tile()), coord(lane_id / 4, (lane_id % 4) * 2));
Coord<CoordSpace<CoordTile2D<MMA_M, MMA_N>>> thread_smem_coord_1 = thread_coord_map(r2s_smem_coord_space.step(offset(8, 0), unit_tile()), coord(lane_id / 4, (lane_id % 4) * 2));

thread_copy(half, 32, "r2s", get_crd_ptr(RC, thread_reg_coord_0), get_crd_ptr(s_c, thread_smem_coord_0));
thread_copy(half, 32, "r2s", get_crd_ptr(RC, thread_reg_coord_1), get_crd_ptr(s_c, thread_smem_coord_1));
```

### Pattern 4: Shared-to-Global Copy (S2G) - Thread-Level

**Before (Original):**
```cuda
if (lane_id < MMA_M) {  // Manual boundary check
  int store_gmem_addr = store_gmem_m * N + store_gmem_n;
  thread_copy(half, 128, "s2g", &s_c[lane_id][0], &C[store_gmem_addr]);
}
```

**After (Fully Automatic Boundary Inference - LATEST):**
```cuda
CoordTile2D<MMA_M, MMA_N> s2g_tile = tile2d<MMA_M, MMA_N>();
CoordSpace<CoordTile2D<MMA_M, MMA_N>> s2g_coord_space = make_coord_space(s2g_tile);
CoordSpace<CoordTile2D<MMA_M, MMA_N>> s2g_smem_space = s2g_coord_space.step(offset(0, 0), s2g_tile);
CoordSpace<CoordTile2D<MMA_M, MMA_N>> s2g_gmem_space = s2g_coord_space.step(offset(by, bx), s2g_tile);

// No explicit boundary checks needed - DSL handles everything automatically!
Coord<CoordSpace<CoordTile2D<MMA_M, MMA_N>>> thread_smem_coord = thread_coord_map(s2g_smem_space, coord(lane_id, _));
Coord<CoordSpace<CoordTile2D<MMA_M, MMA_N>>> thread_gmem_coord = thread_coord_map(s2g_gmem_space, coord(lane_id, _));
PtrOffset gmem_offset = make_ptr_offset(thread_gmem_coord, stride(N, 1));
thread_copy(half, 128, "s2g", get_crd_ptr(s_c, thread_smem_coord), get_off_ptr(C, gmem_offset));
// DSL automatically ensures: lane_id >= MMA_M threads access safe dummy storage
```

**Alternative (Legacy - Manual Boundary Check):**
```cuda
if (lane_id < MMA_M) {  // Manual preservation of original check
  CoordTile2D<MMA_M, MMA_N> s2g_tile = tile2d<MMA_M, MMA_N>();
  // ... rest of coordinate setup ...
  thread_copy(half, 128, "s2g", get_crd_ptr(s_c, thread_smem_coord), get_off_ptr(C, gmem_offset));
}
```

## Common Transformation Patterns

### Pattern A: Linear Memory Access (1D Operations)
```cuda
// Original
int idx = threadIdx.x + blockIdx.x * blockDim.x;
output[idx] = input[idx];

// Transformed  
CoordTile1D<BLOCK_SIZE> tile = tile1d<BLOCK_SIZE>();
CoordSpace<CoordTile1D<BLOCK_SIZE>> coord_space = make_coord_space(tile);
CoordSpace<CoordTile1D<BLOCK_SIZE>> global_space = coord_space.step(offset(blockIdx.x), tile);
Coord<CoordSpace<CoordTile1D<BLOCK_SIZE>>> thread_coord = thread_coord_map(global_space, coord(threadIdx.x));
PtrOffset offset_val = make_ptr_offset(thread_coord, stride(1));
get_off_ptr(output, offset_val)[0] = get_off_ptr(input, offset_val)[0];
```

### Pattern B: 2D Tiled Access (Matrix Operations)
```cuda
// Original
int row = blockIdx.y * TILE_M + threadIdx.y;
int col = blockIdx.x * TILE_N + threadIdx.x;
output[row * N + col] = input[row * K + col];

// Transformed
CoordTile2D<TILE_M, TILE_N> tile = tile2d<TILE_M, TILE_N>();  
CoordSpace<CoordTile2D<TILE_M, TILE_N>> coord_space = make_coord_space(tile);
CoordSpace<CoordTile2D<TILE_M, TILE_N>> output_space = coord_space.step(offset(blockIdx.y, blockIdx.x), tile);
Coord<CoordSpace<CoordTile2D<TILE_M, TILE_N>>> thread_coord = thread_coord_map(output_space, coord(threadIdx.y, threadIdx.x));
PtrOffset output_offset = make_ptr_offset(thread_coord, stride(N, 1));
PtrOffset input_offset = make_ptr_offset(thread_coord, stride(K, 1));
get_off_ptr(output, output_offset)[0] = get_off_ptr(input, input_offset)[0];
```

### Pattern C: Shared Memory Coordination
```cuda
// Original
__shared__ float smem[TILE_M][TILE_N];
smem[threadIdx.y][threadIdx.x] = global_data[...];

// Transformed
mem_alloc_shared(float, smem, [TILE_M][TILE_N]);
CoordTile2D<TILE_M, TILE_N> smem_tile = tile2d<TILE_M, TILE_N>();
CoordSpace<CoordTile2D<TILE_M, TILE_N>> smem_space = make_coord_space(smem_tile);
Coord<CoordSpace<CoordTile2D<TILE_M, TILE_N>>> thread_smem_coord = thread_coord_map(smem_space, coord(threadIdx.y, threadIdx.x));
get_crd_ptr(smem, thread_smem_coord)[0] = global_data[...];
```

### Step 4: Preserve All Other Operations
- Keep `mma(...)` operations unchanged
- Maintain `block_sync()` calls at identical positions  
- Preserve `serial_range_for(...)` loop structures
- Keep all memory allocation (`mem_alloc_shared`, `mem_alloc_register`) unchanged

## Algorithm-Specific Guidelines

### GEMM Operations
- **A/B matrix tiles**: Use `CoordTile2D<M, K>` and `CoordTile2D<K, N>`
- **C matrix tiles**: Use `CoordTile2D<M, N>` 
- **Register tiles**: Use `CoordTile2D<2, 2>` for half-precision registers

### Convolution Operations  
- **Input tiles**: Use `CoordTile3D<C, H, W>` or `CoordTile2D<H, W>`
- **Weight tiles**: Use `CoordTile4D<C_OUT, C_IN, KH, KW>`
- **Sliding window**: Use `step(offset(...), unit_tile())` for stride/dilation

### Reduction Operations
- **Vector reductions**: Use `CoordTile1D<N>` for input vectors
- **Warp reductions**: Use appropriate warp-level coordinate mapping
- **Hierarchical reductions**: Use smaller tiles for partial results

## Important Considerations

### 1. Preserve Original Logic
- **CRITICAL**: Keep identical synchronization points (`block_sync()`)
- **CRITICAL**: Maintain ALL boundary checks and conditional execution (e.g., `if (lane_id < MMA_M)`, `if (tid < LIMIT)`)
- **CRITICAL**: Preserve thread participation patterns - not all threads in a warp may participate in every operation
- Preserve register allocation and variable lifetime
- Keep the same algorithmic flow and data dependencies

### 🚀 NEW: Fully Automatic Boundary Inference (LATEST)
**The DSL now provides COMPLETE automatic boundary inference** - no explicit boundary checks needed at all!

```cuda
// LATEST: Fully automatic boundary management - no explicit checks needed!
CoordTile2D<MMA_M, MMA_N> tile = tile2d<MMA_M, MMA_N>();
CoordSpace<CoordTile2D<MMA_M, MMA_N>> space = make_coord_space(tile);

// DSL automatically infers lane_id < MMA_M and handles safety internally
auto thread_coord = thread_coord_map(space, coord(lane_id, _));
thread_copy(...); // Completely automatic and safe - no boundary checks needed!
```

**Key Benefits:**
- ✅ **Zero API Changes**: Existing code works unchanged
- ✅ **100% Safety**: Invalid threads access safe dummy storage  
- ✅ **Perfect Performance**: Same efficiency as manual checks
- ✅ **Automatic Correctness**: Impossible to forget boundary checks

### ⚠️ Legacy Approaches (No Longer Needed)
**Manual boundary management (OLD):**
```cuda
// OLD: Manual boundary checking (no longer needed)
if (lane_id < MMA_M) {
  CoordTile2D<MMA_M, MMA_N> tile = tile2d<MMA_M, MMA_N>();
  thread_copy(...); // Manual safety check
}

// SEMI-AUTO: Semi-automatic with explicit check (no longer needed)  
if (COORD_BOUNDARY_CHECK(space, coord(lane_id, _))) {
  thread_copy(...); // Explicit check required
}
```

**📚 See [Fully Automatic Boundary Inference](./fully_automatic_boundary_inference.md) for complete documentation.**

### 2. Type Specifications
- Always use explicit coordinate types (never `auto`)
- Match coordinate tile dimensions to algorithm requirements  
- Use appropriate granularity (thread/warp/block level)

### 3. Memory Access Patterns
- Ensure coalesced global memory access is maintained
- Preserve shared memory banking patterns
- Keep register usage comparable to original

## Quality Requirements

1. **Functional Equivalence**: Transformed code must produce identical results
2. **Performance Preservation**: No performance regression from conversion  
3. **Type Safety**: All coordinate types explicitly specified
4. **Code Clarity**: Coordinate mappings should be understandable
5. **Maintainability**: Use consistent naming and patterns

## Validation Checklist

After conversion, verify:
- [ ] All memory access patterns use coordinate-based abstractions
- [ ] Original algorithm logic and semantics preserved
- [ ] **NEW**: Boundary checks can be completely removed - DSL handles automatically
- [ ] **OPTIONAL**: Manual boundary checks preserved if desired (backward compatibility)
- [ ] All `block_sync()` calls maintained at original positions
- [ ] Coordinate tile dimensions match algorithm requirements
- [ ] Global memory access patterns remain coalesced
- [ ] No `auto` type deduction used for coordinates
- [ ] Code compiles successfully with both headers included
- [ ] Runtime correctness verified against original kernel
- [ ] **NEW**: Verify automatic boundary inference works (lane_id >= MMA_M threads should be safe)

## Usage Instructions

**PRIMARY REFERENCE**: Always consult the [Coordinate DSL API Reference](./coordinate_dsl_api_reference.md) for precise API syntax, parameters, and complete usage examples.

**TRANSFORMATION WORKFLOW**:
1. **Study the API Reference** to understand all coordinate DSL operations
2. **Identify all memory copy operations** in the provided kernel  
3. **Apply the appropriate transformation patterns** from the API reference
4. **Verify coordinate tile dimensions** match algorithm requirements
5. **Ensure all coordinate types are explicit** (no `auto` usage)
6. **Test the transformed code** against the validation checklist

**FOCUS**: Create maintainable, type-safe coordinate abstractions that preserve the original kernel's functionality and performance while providing a higher level of abstraction for memory access patterns.
**FOCUS**: Create maintainable, type-safe coordinate abstractions that preserve the original kernel's functionality and performance while providing a higher level of abstraction for memory access patterns.