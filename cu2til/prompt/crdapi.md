# Coordinate-based DSL API Reference

## Overview

This document provides a comprehensive reference for all coordinate-based DSL APIs. Each API is precisely defined with syntax, parameters, usage examples, and semantic constraints.

## Quick Reference Index

| Operation Category | APIs | Section |
|-------------------|------|---------|
| **Memory Copy** | `thread_copy`, `warp_copy` | [Memory Copy Operations](#memory-copy-operations) |
| **Coordinate Tiles** | `tile1d`, `tile2d`, `tile3d`, `tile4d`, `mma_tile`, `warp_tile`, `reg_tile`, `unit_tile` | [Coordinate System APIs](#coordinate-system-apis) |
| **Coordinate Spaces** | `make_coord_space`, `.step()`, `thread_coord_map` | [Coordinate System APIs](#coordinate-system-apis) |
| **Coordinate Creation** | `coord`, `offset`, `stride` | [Coordinate System APIs](#coordinate-system-apis) |
| **Memory Addressing** | `make_ptr_offset`, `get_crd_ptr`, `get_off_ptr` | [Memory Address Calculation](#memory-address-calculation) |
| **Compute** | `mma` | [Compute Operations](#compute-operations) |
| **Synchronization** | `block_sync` | [Synchronization Operations](#synchronization-operations) |
| **Control Flow** | `serial_range_for` | [Control Flow Operations](#control-flow-operations) |
| **Memory Allocation** | `mem_alloc_shared`, `mem_alloc_register`, `mem_fill` | [Memory Allocation Operations](#memory-allocation-operations) |

## Memory Copy Quick Reference

| Source | Destination | API | Execution Model |
|--------|-------------|-----|-----------------|
| Global | Shared | `thread_copy(dtype, bits, "g2s", ...)` | Single thread |
| Shared | Global | `thread_copy(dtype, bits, "s2g", ...)` | Single thread |
| Global | Register | `thread_copy(dtype, bits, "g2r", ...)` | Single thread |
| Register | Global | `thread_copy(dtype, bits, "r2g", ...)` | Single thread |
| Shared | Register | `thread_copy(dtype, bits, "s2r", ...)` OR `warp_copy(dtype, bits, "s2r", layout, ...)` | Single thread OR Warp collective |
| Register | Shared | `thread_copy(dtype, bits, "r2s", ...)` | Single thread |

---

## Memory Copy Operations

Memory copy operations are categorized by **execution model** rather than memory types, as this better reflects the underlying hardware capabilities.

### Thread-Level Copy (`thread_copy`)

**Execution Model**: Single thread operation  
**Supported Memory Types**: Global (G) ↔ Shared (S) ↔ Register (R) (any combination)  
**Underlying Hardware**: Standard load/store instructions with vectorization  

#### Syntax
```cuda
thread_copy(dtype, bits, "task", src_ptr, dst_ptr);
```

#### Parameters
- **`dtype`**: Data type (`half`, `float`, `int`, `uint32_t`, etc.)
- **`bits`**: Transfer width (`32`, `64`, `128` bits)
- **`task`**: Transfer direction string
- **`src_ptr`**: Source pointer (from `get_crd_ptr()` or `get_off_ptr()`)
- **`dst_ptr`**: Destination pointer (from `get_crd_ptr()` or `get_off_ptr()`)

#### Task Strings
| Task | Direction | Description |
|------|-----------|-------------|
| `"g2s"` | Global → Shared | Load from global memory to shared memory |
| `"s2g"` | Shared → Global | Store from shared memory to global memory |
| `"g2r"` | Global → Register | Load from global memory to registers |
| `"r2g"` | Register → Global | Store from registers to global memory |
| `"s2r"` | Shared → Register | Load from shared memory to registers |
| `"r2s"` | Register → Shared | Store from registers to shared memory |

#### Transfer Width Requirements
| Bits | Data Types | Elements | Hardware Implementation |
|------|------------|----------|-------------------------|
| 32 | `half` | 2 elements | `half2` load/store |
| 64 | `half` | 4 elements | `float2` reinterpret cast |
| 128 | `half` | 8 elements | `float4` reinterpret cast |
| 32 | `float` | 1 element | Direct `float` load/store |
| 64 | `float` | 2 elements | `float2` load/store |
| 128 | `float` | 4 elements | `float4` load/store |

#### Comprehensive Usage Examples

**G2S (Global to Shared) - Complete Flow:**
```cuda
// 1. Define coordinate tiles and spaces
CoordTile2D<MMA_M, MMA_K> g2s_tile = tile2d<MMA_M, MMA_K>();
CoordSpace<CoordTile2D<MMA_M, MMA_K>> g2s_coord_space = make_coord_space(g2s_tile);

// 2. Create offset coordinate spaces for global and shared memory
CoordSpace<CoordTile2D<MMA_M, MMA_K>> g2s_gmem_space = g2s_coord_space.step(offset(by, k), g2s_tile);
CoordSpace<CoordTile2D<MMA_M, MMA_K>> g2s_smem_space = g2s_coord_space.step(offset(0, 0), g2s_tile);

// 3. Map thread to coordinate space
Coord<CoordSpace<CoordTile2D<MMA_M, MMA_K>>> thread_g2s_gmem_coord = thread_coord_map(g2s_gmem_space, coord(tid/2, (tid%2)*8));
Coord<CoordSpace<CoordTile2D<MMA_M, MMA_K>>> thread_g2s_smem_coord = thread_coord_map(g2s_smem_space, coord(tid/2, (tid%2)*8));

// 4. Calculate linear offset for global memory and perform copy
PtrOffset g2s_gmem_off = make_ptr_offset(thread_g2s_gmem_coord, stride(K, 1));
thread_copy(half, 128, "g2s", get_off_ptr(A, g2s_gmem_off), get_crd_ptr(s_a, thread_g2s_smem_coord));
```

**G2R (Global to Register) - Direct Access:**
```cuda
// For direct global to register copies (less common)
CoordTile1D<VECTOR_SIZE> g2r_tile = tile1d<VECTOR_SIZE>();
CoordSpace<CoordTile1D<VECTOR_SIZE>> g2r_coord_space = make_coord_space(g2r_tile);
CoordSpace<CoordTile1D<VECTOR_SIZE>> g2r_gmem_space = g2r_coord_space.step(offset(block_offset), g2r_tile);

Coord<CoordSpace<CoordTile1D<VECTOR_SIZE>>> thread_g2r_gmem_coord = thread_coord_map(g2r_gmem_space, coord(thread_offset));
PtrOffset g2r_gmem_off = make_ptr_offset(thread_g2r_gmem_coord, stride(1));
thread_copy(float, 128, "g2r", get_off_ptr(input, g2r_gmem_off), reg_ptr);
```

**R2S (Register to Shared):**
```cuda
CoordTile2D<2, 2> reg_tile = reg_tile<2, 2>();
CoordTile2D<MMA_M, MMA_N> smem_tile = tile2d<MMA_M, MMA_N>();
CoordSpace<CoordTile2D<2, 2>> reg_space = make_coord_space(reg_tile);
CoordSpace<CoordTile2D<MMA_M, MMA_N>> smem_space = make_coord_space(smem_tile);

Coord<CoordSpace<CoordTile2D<2, 2>>> thread_reg_coord = thread_coord_map(reg_space.step(offset(0, 0), unit_tile()), coord(0, _));
Coord<CoordSpace<CoordTile2D<MMA_M, MMA_N>>> thread_smem_coord = thread_coord_map(smem_space.step(offset(0, 0), unit_tile()), coord(lane_id/4, (lane_id%4)*2));

thread_copy(half, 32, "r2s", get_crd_ptr(RC, thread_reg_coord), get_crd_ptr(s_c, thread_smem_coord));
```

### Warp-Level Copy (`warp_copy`)

**Execution Model**: Warp-collective operation (32 threads cooperate)  
**Primary Use Case**: Shared Memory → Register with `ldmatrix` instruction  
**Hardware Requirement**: Requires warp-synchronous execution  

#### Syntax
```cuda
warp_copy(dtype, bits, "task", layout, src_ptr, dst_reg0, dst_reg1, [dst_reg2, dst_reg3]);
```

#### Parameters  
- **`dtype`**: Data type (typically `uint32_t` for ldmatrix)
- **`bits`**: Transfer width (`64` for 2 registers, `128` for 4 registers)
- **`task`**: Transfer direction (typically `"s2r"`)
- **`layout`**: Matrix layout character (`'n'` = row-major, `'t'` = column-major/transposed)
- **`src_ptr`**: Source pointer from shared memory
- **`dst_reg0, dst_reg1, ...`**: Destination register array elements

#### Task Strings for Warp Copy
| Task | Direction | Description | Hardware |
|------|-----------|-------------|----------|
| `"s2r"` | Shared → Register | Warp-collective load using `ldmatrix` | `ldmatrix.sync.aligned` |

#### Layout Characters
| Layout | Matrix Interpretation | ldmatrix Variant |
|--------|----------------------|------------------|
| `'n'` | Row-major (non-transposed) | `ldmatrix.x2/x4` |
| `'t'` | Column-major (transposed) | `ldmatrix.x2.trans/x4.trans` |

#### Usage Examples

**S2R with 4 Registers (128-bit):**
```cuda
CoordTile2D<16, 8> warp_tile = warp_tile<16, 8>();
CoordSpace<CoordTile2D<16, 8>> smem_space = make_coord_space(warp_tile);
CoordSpace<CoordTile2D<16, 8>> smem_coord_space = smem_space.step(offset(0, 0), warp_tile);
Coord<CoordSpace<CoordTile2D<16, 8>>> thread_smem_coord = thread_coord_map(smem_coord_space, coord(lane_id % 16, (lane_id / 16) * 8));

warp_copy(uint32_t, 128, "s2r", 'n', get_crd_ptr(s_a, thread_smem_coord), RA[0], RA[1], RA[2], RA[3]);
```

**S2R with 2 Registers (64-bit, Transposed):**
```cuda
CoordTile2D<16, 8> warp_tile = warp_tile<16, 8>();
CoordSpace<CoordTile2D<16, 8>> smem_space = make_coord_space(warp_tile);
CoordSpace<CoordTile2D<16, 8>> smem_coord_space = smem_space.step(offset(0, 0), warp_tile);
Coord<CoordSpace<CoordTile2D<16, 8>>> thread_smem_coord = thread_coord_map(smem_coord_space, coord(lane_id % 16, 0));

warp_copy(uint32_t, 64, "s2r", 't', get_crd_ptr(s_b, thread_smem_coord), RB[0], RB[1]);
```

---

## Coordinate System APIs

### Coordinate Tile Creation

#### Multi-dimensional Tile Factories
```cuda
// 1D tiles
CoordTile1D<N> tile1d<N>();
CoordTile1D<1024> vector_tile = tile1d<1024>();

// 2D tiles  
CoordTile2D<M, N> tile2d<M, N>();
CoordTile2D<16, 16> matrix_tile = tile2d<16, 16>();

// 3D tiles
CoordTile3D<D, H, W> tile3d<D, H, W>();
CoordTile3D<8, 32, 32> tensor_tile = tile3d<8, 32, 32>();

// 4D tiles
CoordTile4D<N, C, H, W> tile4d<N, C, H, W>();
CoordTile4D<4, 256, 14, 14> feature_tile = tile4d<4, 256, 14, 14>();
```

#### Semantic Tile Factories
```cuda
// Common pattern factories (parameterized)
template<int M, int N> auto mma_tile() -> CoordTile2D<M, N>;
template<int M, int N> auto warp_tile() -> CoordTile2D<M, N>;  
template<int M, int N> auto reg_tile() -> CoordTile2D<M, N>;

// Unit tile for element-level operations
auto unit_tile() -> CoordTile2D<1, 1>;

// Usage examples
CoordTile2D<16, 8> warp_tile = warp_tile<16, 8>();
CoordTile2D<2, 2> register_tile = reg_tile<2, 2>();
CoordTile2D<1, 1> unit_tile = unit_tile();
```

### Coordinate Space Management

#### Space Creation
```cuda
template<typename TileType>
CoordSpace<TileType> make_coord_space(TileType tile);

// Example
CoordTile2D<16, 16> tile = tile2d<16, 16>();
CoordSpace<CoordTile2D<16, 16>> coord_space = make_coord_space(tile);
```

#### Space Transformation
```cuda
template<typename TileType>
CoordSpace<TileType> CoordSpace<TileType>::step(offset_t offset, TileType step_tile);

// Example
CoordSpace<CoordTile2D<16, 16>> global_space = coord_space.step(offset(by, bx), tile);
CoordSpace<CoordTile2D<16, 16>> local_space = coord_space.step(offset(0, 0), tile);
```

#### Thread Coordinate Mapping
```cuda
template<typename CoordSpaceType, typename CoordType>
Coord<CoordSpaceType> thread_coord_map(CoordSpaceType space, CoordType thread_coord);

// Examples
Coord<CoordSpace<CoordTile2D<16, 16>>> thread_coord = thread_coord_map(space, coord(tid_x, tid_y));
Coord<CoordSpace<CoordTile2D<16, 16>>> full_row_coord = thread_coord_map(space, coord(row, _));
```

### Coordinate Creation

#### Basic Coordinate Factory
```cuda
template<typename T1, typename T2>
coord_t<T1, T2> coord(T1 m, T2 n);

// Examples
auto point_coord = coord(5, 10);        // Specific point (5, 10)
auto row_coord = coord(3, _);           // Full row 3
auto col_coord = coord(_, 7);           // Full column 7
```

#### Offset Creation
```cuda
template<typename T1, typename T2>  
offset_t<T1, T2> offset(T1 m, T2 n);

// Examples
auto block_offset = offset(blockIdx.y, blockIdx.x);
auto zero_offset = offset(0, 0);
auto step_offset = offset(by * MMA_M, bx * MMA_N);
```

#### Stride Creation
```cuda
template<typename T1, typename T2>
stride_t<T1, T2> stride(T1 m_stride, T2 n_stride);

// Examples  
auto row_major_stride = stride(N, 1);       // Row-major matrix with N columns
auto col_major_stride = stride(1, M);       // Column-major matrix with M rows
auto custom_stride = stride(K, 1);          // Custom stride pattern
```

---

## Memory Address Calculation

### Pointer Offset Calculation
```cuda
template<typename ThreadCoordType, typename StrideType>
PtrOffset make_ptr_offset(ThreadCoordType thread_coord, StrideType mem_stride);

// Example
PtrOffset global_offset = make_ptr_offset(thread_global_coord, stride(K, 1));
```

### Pointer Retrieval

#### Coordinate-based Pointer Access
```cuda
// For 2D arrays (shared memory)
template<typename T, typename ThreadCoordType>
T* get_crd_ptr(T array[][N], ThreadCoordType thread_coord);

// For 1D arrays (registers) 
template<typename T, typename ThreadCoordType>
T* get_crd_ptr(T array[], ThreadCoordType thread_coord);

// Examples
half* smem_ptr = get_crd_ptr(s_a, thread_smem_coord);
uint32_t* reg_ptr = get_crd_ptr(RC, thread_reg_coord);
```

#### Offset-based Pointer Access  
```cuda
template<typename T>
T* get_off_ptr(T* base_ptr, PtrOffset offset);

// Example
half* global_ptr = get_off_ptr(A, global_offset);
```

---

## Compute Operations

### Matrix Multiply-Accumulate (MMA)

#### Syntax
```cuda
mma(shape, layout, input_type, output_type, 
    dst0, dst1, 
    src_a0, src_a1, src_a2, src_a3, 
    src_b0, src_b1, 
    src_c0, src_c1);
```

#### Parameters
- **`shape`**: MMA instruction shape (`"m16n8k16"`, `"m16n16k16"`, etc.)
- **`layout`**: Matrix layouts (`"tn"` = A transposed, B non-transposed)  
- **`input_type`**: Input data type (`f16`, `bf16`, `s8`, etc.)
- **`output_type`**: Output data type (`f16`, `f32`, `s32`, etc.)
- **`dst0, dst1`**: Destination accumulator registers
- **`src_a0-src_a3`**: Source A matrix registers (4 registers)
- **`src_b0, src_b1`**: Source B matrix registers (2 registers)  
- **`src_c0, src_c1`**: Source C accumulator registers

#### Layout Strings
| Layout | A Matrix | B Matrix | PTX Equivalent |
|--------|----------|----------|----------------|
| `"tn"` | Transposed (col-major) | Non-transposed (row-major) | `row.col` |
| `"nt"` | Non-transposed (row-major) | Transposed (col-major) | `col.row` |

#### Usage Example
```cuda
// Perform MMA operation: C = A * B + C
mma("m16n8k16", "tn", f16, f16, 
    RC[0], RC[1],                    // Destination/accumulator 
    RA[0], RA[1], RA[2], RA[3],     // A matrix fragments
    RB[0], RB[1],                   // B matrix fragments  
    RC[0], RC[1]);                  // Source accumulator
```

---

## Synchronization Operations

### Block Synchronization
```cuda
block_sync();
```
**Equivalent to**: `__syncthreads()`  
**Semantics**: All threads in the block reach this point before any continue

---

## Control Flow Operations

### Range-Based Serial Loop
```cuda
serial_range_for(iterator, start, end, step) {
    // loop body
}
```

**Parameters**:
- **`iterator`**: Loop variable name
- **`start`**: Starting value  
- **`end`**: Ending value (exclusive)
- **`step`**: Step size

**Equivalent to**:
```cuda
#pragma unroll
for (int iterator = start; iterator < end; iterator += step) {
    // loop body
}
```

**Usage Example**:
```cuda
serial_range_for(k, 0, NUM_K_TILES, 1) {
    // Process k-th tile
    // ... memory operations ...
    block_sync();
}
```

---

## Memory Allocation Operations

### Shared Memory Allocation
```cuda
mem_alloc_shared(dtype, var_name, [dim1][dim2]...);
```

**Parameters**:
- **`dtype`**: Data type (`half`, `float`, `int`, etc.)
- **`var_name`**: Variable name
- **`[dim1][dim2]...`**: Multi-dimensional array dimensions

**Equivalent to**: `__shared__ dtype var_name[dim1][dim2]...;`

**Usage Examples**:
```cuda
mem_alloc_shared(half, s_a, [MMA_M][MMA_K]);       // 2D shared array
mem_alloc_shared(float, s_temp, [256]);             // 1D shared array
```

### Register Allocation
```cuda
mem_alloc_register(dtype, var_name, [size]);
```

**Parameters**:
- **`dtype`**: Data type
- **`var_name`**: Variable name  
- **`[size]`**: Array size

**Equivalent to**: `dtype var_name[size];`

**Usage Examples**:
```cuda
mem_alloc_register(uint32_t, RA, [4]);              // 4 uint32_t registers
mem_alloc_register(half, temp, [8]);                // 8 half registers
```

### Memory Initialization
```cuda
mem_fill(array_name, init_value);
```

**Parameters**:
- **`array_name`**: Array variable name
- **`init_value`**: Initialization value

**Usage Example**:
```cuda
mem_alloc_register(uint32_t, RC, [2]);
mem_fill(RC, 0u);                                   // Initialize to zero
```

---

## Type System

### DSL Type Aliases
```cuda
TunableInt      // For tunable parameters (MMA_M, MMA_N, MMA_K)
ShapeInt        // For shape constants (M, N, K)
IndexInt        // For loop indices and calculated values  
ArchInt         // For architecture constants (WARP_SIZE)
PtrInt          // For pointer conversion (uint32_t addresses)
PtrOffset       // For linear memory offsets (int)
```

### Usage Guidelines
- Use **explicit coordinate types** instead of `auto`
- Match coordinate tile dimensions to algorithm requirements
- Choose appropriate granularity (thread/warp/block level)

---

## Common Usage Patterns

### Pattern 1: Complete G2S Flow
```cuda
// 1. Define tiles and spaces
CoordTile2D<TILE_M, TILE_K> tile = tile2d<TILE_M, TILE_K>();
CoordSpace<CoordTile2D<TILE_M, TILE_K>> coord_space = make_coord_space(tile);

// 2. Create global and shared memory spaces  
CoordSpace<CoordTile2D<TILE_M, TILE_K>> gmem_space = coord_space.step(offset(block_row, block_col), tile);
CoordSpace<CoordTile2D<TILE_M, TILE_K>> smem_space = coord_space.step(offset(0, 0), tile);

// 3. Map thread coordinates
Coord<CoordSpace<CoordTile2D<TILE_M, TILE_K>>> thread_gmem_coord = thread_coord_map(gmem_space, coord(local_row, local_col));
Coord<CoordSpace<CoordTile2D<TILE_M, TILE_K>>> thread_smem_coord = thread_coord_map(smem_space, coord(local_row, local_col));

// 4. Calculate pointer offsets and perform copy
PtrOffset gmem_offset = make_ptr_offset(thread_gmem_coord, stride(STRIDE, 1));
thread_copy(dtype, bits, "g2s", get_off_ptr(global_array, gmem_offset), get_crd_ptr(shared_array, thread_smem_coord));
```

### Pattern 2: Complete S2R Warp Copy Flow
```cuda
// 1. Define warp tile
CoordTile2D<16, 8> warp_tile = warp_tile<16, 8>();
CoordSpace<CoordTile2D<16, 8>> warp_space = make_coord_space(warp_tile);

// 2. Create shared memory space
CoordSpace<CoordTile2D<16, 8>> smem_space = warp_space.step(offset(0, 0), warp_tile);

// 3. Map lane to coordinate
Coord<CoordSpace<CoordTile2D<16, 8>>> thread_coord = thread_coord_map(smem_space, coord(lane_id % 16, (lane_id / 16) * 8));

// 4. Perform warp copy
warp_copy(uint32_t, 128, "s2r", 'n', get_crd_ptr(shared_array, thread_coord), reg[0], reg[1], reg[2], reg[3]);
```

---

## Error Prevention Guidelines

### Common Mistakes to Avoid

1. **Using `auto` for coordinate types**
   ```cuda
   // WRONG
   auto tile = tile2d<16, 16>();
   
   // CORRECT  
   CoordTile2D<16, 16> tile = tile2d<16, 16>();
   ```

2. **Mismatching tile dimensions**
   ```cuda
   // WRONG: Using MMA tile for register operations
   CoordTile2D<16, 16> reg_tile = tile2d<16, 16>();
   
   // CORRECT: Using appropriate register tile
   CoordTile2D<2, 2> reg_tile = reg_tile<2, 2>();
   ```

3. **Wrong memory copy API selection**
   ```cuda
   // WRONG: Using warp_copy for G2S
   warp_copy(half, 128, "g2s", 'n', ...);
   
   // CORRECT: Using thread_copy for G2S
   thread_copy(half, 128, "g2s", ...);
   ```

4. **Incorrect stride calculation**
   ```cuda
   // WRONG: Column-major stride for row-major data
   stride(1, N)
   
   // CORRECT: Row-major stride  
   stride(N, 1)
   ```

### Debugging Tips

1. **Check coordinate tile dimensions** match your algorithm's blocking strategy
2. **Verify stride patterns** match your memory layout (row-major vs column-major)  
3. **Ensure thread coordinate mapping** matches your indexing pattern
4. **Confirm memory copy API usage** matches the hardware capabilities
5. **Validate synchronization points** are preserved from original code

---

This API reference provides precise specifications for all coordinate-based DSL operations, enabling accurate code generation and transformation.