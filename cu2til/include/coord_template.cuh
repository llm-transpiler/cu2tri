#pragma once

#include "dsl_template.cuh"

// ========================= Coordinate System Types =========================

// Pointer offset type for flexible addressing
using PtrOffset = int;

// Stride structure for memory layout description
template<typename T1, typename T2>
struct stride_t {
    T1 m_stride;
    T2 n_stride;
    
    __device__ __host__ constexpr stride_t(T1 m_val, T2 n_val) : m_stride(m_val), n_stride(n_val) {}
};

// ========================= Multi-Dimensional Coordinate Tile System =========================

// Forward declaration for multi-dimensional coordinate tiles
template<int... Dims>
struct crd_tile_t;

// 1D coordinate tile specialization
template<int N>
struct crd_tile_t<N> {
    static constexpr int dim_0 = N;
    static constexpr int rank = 1;
    
    __device__ __host__ constexpr crd_tile_t() {}
};

// 2D coordinate tile specialization (most common for GPU kernels)
template<int M, int N>
struct crd_tile_t<M, N> {
    static constexpr int dim_0 = M;  // First dimension (rows)
    static constexpr int dim_1 = N;  // Second dimension (columns)
    static constexpr int rank = 2;
    
    // Backward compatibility aliases
    static constexpr int dim_m = M;
    static constexpr int dim_n = N;
    
    __device__ __host__ constexpr crd_tile_t() {}
};

// 3D coordinate tile specialization  
template<int M, int N, int K>
struct crd_tile_t<M, N, K> {
    static constexpr int dim_0 = M;
    static constexpr int dim_1 = N;
    static constexpr int dim_2 = K;
    static constexpr int rank = 3;
    
    __device__ __host__ constexpr crd_tile_t() {}
};

// 4D coordinate tile specialization
template<int M, int N, int K, int L>
struct crd_tile_t<M, N, K, L> {
    static constexpr int dim_0 = M;
    static constexpr int dim_1 = N;
    static constexpr int dim_2 = K;
    static constexpr int dim_3 = L;
    static constexpr int rank = 4;
    
    __device__ __host__ constexpr crd_tile_t() {}
};

// Coordinate point structure
template<typename T1, typename T2>
struct coord_t {
    T1 m;
    T2 n;
    
    __device__ __host__ constexpr coord_t(T1 m_val, T2 n_val) : m(m_val), n(n_val) {}
};

// Special coordinate marker for accessing full dimension
struct coord_full_dim_t {};
static constexpr coord_full_dim_t _ {};

// Offset structure
template<typename T1, typename T2>
struct offset_t {
    T1 m;
    T2 n;
    
    __device__ __host__ constexpr offset_t(T1 m_val, T2 n_val) : m(m_val), n(n_val) {}
};

// Coordinate space structure
template<typename TileType>
struct coord_space_t {
    TileType tile;
    int offset_m;
    int offset_n;
    
    __device__ __host__ constexpr coord_space_t(TileType t, int om = 0, int on = 0) 
        : tile(t), offset_m(om), offset_n(on) {}
    
    // Step method for coordinate space transformation
    template<typename OffsetType, typename StepTileType>
    __device__ __host__ constexpr auto step(OffsetType offset, StepTileType step_tile) const {
        return coord_space_t<TileType>(
            tile,
            offset_m + offset.m * step_tile.dim_m,
            offset_n + offset.n * step_tile.dim_n
        );
    }
};

// Enhanced thread coordinate structure with automatic boundary inference
template<typename CoordType>
struct thread_coord_t {
    CoordType base_coord;
    int thread_offset_m;
    int thread_offset_n;
    bool is_full_dim_n;
    bool is_within_bounds;  // NEW: Automatic boundary check result
    
    template<typename T1, typename T2>
    __device__ __host__ constexpr thread_coord_t(CoordType base, coord_t<T1, T2> thread_coord)
        : base_coord(base), thread_offset_m(thread_coord.m), thread_offset_n(thread_coord.n), is_full_dim_n(false) {
        // Automatic boundary inference: check if thread coordinates are within tile bounds
        #ifdef __CUDA_ARCH__
        is_within_bounds = (thread_offset_m >= 0 && thread_offset_m < base_coord.tile.dim_m) &&
                          (thread_offset_n >= 0 && thread_offset_n < base_coord.tile.dim_n);
        #else
        is_within_bounds = true; // Host compilation - assume valid
        #endif
    }
    
    template<typename T1>
    __device__ __host__ constexpr thread_coord_t(CoordType base, coord_t<T1, coord_full_dim_t> thread_coord)
        : base_coord(base), thread_offset_m(thread_coord.m), thread_offset_n(0), is_full_dim_n(true) {
        // Automatic boundary inference for full dimension access
        #ifdef __CUDA_ARCH__
        is_within_bounds = (thread_offset_m >= 0 && thread_offset_m < base_coord.tile.dim_m);
        // For full dimension (_), n coordinate is always valid
        #else
        is_within_bounds = true; // Host compilation - assume valid
        #endif
    }
    
    // Index access operators for flexible addressing
    __device__ __host__ constexpr int operator[](int idx) const {
        auto& space = base_coord;
        if (idx == 0) {
            return space.offset_m + thread_offset_m;
        } else if (idx == 1) {
            return space.offset_n + thread_offset_n;
        }
        return 0; // fallback
    }
    
    // NEW: Check if this coordinate is within bounds
    __device__ __host__ constexpr bool is_valid() const {
        return is_within_bounds;
    }
};

// ========================= Coordinate System Functions =========================

// ========================= Multi-Dimensional Coordinate Tile Factories =========================

// 1D coordinate tile factory
template<int N>
__device__ __host__ constexpr auto crd_tile() {
    return crd_tile_t<N>{};
}

// 2D coordinate tile factory  
template<int M, int N>
__device__ __host__ constexpr auto crd_tile() {
    return crd_tile_t<M, N>{};
}

// 3D coordinate tile factory
template<int M, int N, int K>
__device__ __host__ constexpr auto crd_tile() {
    return crd_tile_t<M, N, K>{};
}

// 4D coordinate tile factory
template<int M, int N, int K, int L>
__device__ __host__ constexpr auto crd_tile() {
    return crd_tile_t<M, N, K, L>{};
}

// Create coordinate space from tile
template<typename TileType>
__device__ __host__ constexpr auto make_coord_space(TileType tile) {
    return coord_space_t<TileType>(tile);
}

// Create coordinate point
template<typename T1, typename T2>
__device__ __host__ constexpr auto coord(T1 m, T2 n) {
    return coord_t<T1, T2>(m, n);
}

// Create offset
template<typename T1, typename T2>
__device__ __host__ constexpr auto offset(T1 m, T2 n) {
    return offset_t<T1, T2>(m, n);
}

// Create stride
template<typename T1, typename T2>
__device__ __host__ constexpr auto stride(T1 m_stride, T2 n_stride) {
    return stride_t<T1, T2>(m_stride, n_stride);
}

// Thread coordinate mapping
template<typename CoordSpaceType, typename CoordType>
__device__ __host__ constexpr auto thread_coord_map(CoordSpaceType coord_space, CoordType thread_coord) {
    return thread_coord_t<CoordSpaceType>(coord_space, thread_coord);
}

// ========================= Predefined Coordinate Tiles =========================

// Pre-defined unit coordinate tile for element-level operations
static constexpr auto unit_2d_crdtile = crd_tile_t<1, 1>{};

// ========================= Universal Coordinate Type Aliases =========================

// Multi-dimensional coordinate tile aliases (fully generic)
template<int... Dims>
using CoordTile = crd_tile_t<Dims...>;

// Dimension-specific aliases for convenience
template<int N>
using CoordTile1D = CoordTile<N>;

template<int M, int N>
using CoordTile2D = CoordTile<M, N>;

template<int M, int N, int K>
using CoordTile3D = CoordTile<M, N, K>;

template<int M, int N, int K, int L>
using CoordTile4D = CoordTile<M, N, K, L>;

// Universal coordinate space and coordinate aliases
template<typename TileType>
using CoordSpace = coord_space_t<TileType>;

template<typename CoordSpaceType>
using Coord = thread_coord_t<CoordSpaceType>;

// ========================= Convenience Factory Functions =========================

// Multi-dimensional tile factories
template<int N>
inline constexpr auto tile1d() { return crd_tile<N>(); }

template<int M, int N>
inline constexpr auto tile2d() { return crd_tile<M, N>(); }

template<int M, int N, int K>
inline constexpr auto tile3d() { return crd_tile<M, N, K>(); }

template<int M, int N, int K, int L>
inline constexpr auto tile4d() { return crd_tile<M, N, K, L>(); }

// Common pattern factories (parameterized, not fixed)
template<int M, int N>
inline constexpr auto mma_tile() { return crd_tile<M, N>(); }

template<int M, int N>
inline constexpr auto warp_tile() { return crd_tile<M, N>(); }

template<int M, int N>
inline constexpr auto reg_tile() { return crd_tile<M, N>(); }

inline constexpr auto unit_tile() { return crd_tile<1, 1>(); }

// ========================= Usage Examples for Multi-Dimensional Coordinates =========================

/*
// Example 1: 1D coordinate for vector operations
CoordTile1D<1024> vector_tile = tile1d<1024>();
CoordSpace<CoordTile1D<1024>> vector_space = make_coord_space(vector_tile);
Coord<CoordSpace<CoordTile1D<1024>>> thread_vector_coord = thread_coord_map(vector_space, coord(tid));

// Example 2: 2D coordinate for matrix operations (current usage)
CoordTile2D<16, 16> matrix_tile = tile2d<16, 16>();
CoordSpace<CoordTile2D<16, 16>> matrix_space = make_coord_space(matrix_tile);
Coord<CoordSpace<CoordTile2D<16, 16>>> thread_matrix_coord = thread_coord_map(matrix_space, coord(row, col));

// Example 3: 3D coordinate for tensor operations
CoordTile3D<8, 8, 8> tensor_tile = tile3d<8, 8, 8>();
CoordSpace<CoordTile3D<8, 8, 8>> tensor_space = make_coord_space(tensor_tile);
Coord<CoordSpace<CoordTile3D<8, 8, 8>>> thread_tensor_coord = thread_coord_map(tensor_space, coord(x, y, z));

// Example 4: Flexible parameterized usage
template<int M, int N>
void process_matrix() {
    CoordTile2D<M, N> tile = mma_tile<M, N>();    // Parameterized MMA tile
    CoordSpace<CoordTile2D<M, N>> space = make_coord_space(tile);
    Coord<CoordSpace<CoordTile2D<M, N>>> coord = thread_coord_map(space, coord(tid_x, tid_y));
}
*/

// ========================= Memory Address Calculation =========================

// Create pointer offset from coordinate and stride
template<typename ThreadCoordType, typename StrideType>
__device__ __host__ constexpr PtrOffset make_ptr_offset(ThreadCoordType thread_coord, StrideType mem_stride) {
    return thread_coord[0] * mem_stride.m_stride + thread_coord[1] * mem_stride.n_stride;
}

// Enhanced get_crd_ptr with automatic boundary checking for 2D arrays (shared memory)
template<typename TensorType, int M, int N, typename ThreadCoordType>
__device__ auto get_crd_ptr(TensorType (&tensor)[M][N], ThreadCoordType thread_coord) {
    // Automatic boundary checking - if coordinate is invalid, the operation should be safe
    if (!thread_coord.is_valid()) {
        // For invalid coordinates, return a pointer to a safe dummy location
        // This prevents out-of-bounds access while maintaining API compatibility
        static __device__ TensorType dummy_storage = {};
        return &dummy_storage;
    }
    
    auto& space = thread_coord.base_coord;
    int final_m = space.offset_m + thread_coord.thread_offset_m;
    int final_n = space.offset_n + thread_coord.thread_offset_n;
    
    return &tensor[final_m][final_n];
}

// Enhanced register array access with automatic boundary checking
template<typename TensorType, int N, typename ThreadCoordType>
__device__ auto get_crd_ptr(TensorType (&reg_array)[N], ThreadCoordType thread_coord) {
    // Automatic boundary checking for register arrays
    if (!thread_coord.is_valid()) {
        // For invalid coordinates, return a pointer to a safe dummy location
        static __device__ TensorType dummy_storage = {};
        return &dummy_storage;
    }
    
    auto& space = thread_coord.base_coord;
    int final_m = space.offset_m + thread_coord.thread_offset_m;
    // For register arrays, we typically use the m coordinate as the index
    return &reg_array[final_m];
}

// Flexible offset-based pointer access
template<typename TensorType>
__device__ auto get_off_ptr(TensorType* tensor, PtrOffset offset) {
    return &tensor[offset];
}

// ========================= Automatic Boundary Inference System =========================

// Boundary condition extractor - extracts boundary conditions from coordinate usage
template<typename ThreadCoordType>
struct boundary_extractor {
    using coord_space_type = typename ThreadCoordType::base_coord_type;
    using tile_type = typename coord_space_type::tile_type;
    
    // Extract the thread coordinate values that were used
    static constexpr bool has_m_coord = true;  // Always has m coordinate
    static constexpr bool has_n_coord = true;  // May have n coordinate or _
    
    // Get the tile dimensions for boundary checking
    template<int M, int N>
    static constexpr auto get_bounds(crd_tile_t<M, N>) {
        return coord_t<int, int>(M, N);
    }
    
    // Generate boundary condition for a specific thread coordinate
    template<typename CoordSpaceType, typename ThreadCoord>
    __device__ static bool check_bounds(CoordSpaceType coord_space, ThreadCoord thread_coord) {
        // Extract the actual coordinate values
        int thread_m = thread_coord.thread_offset_m;
        int thread_n = thread_coord.thread_offset_n;
        
        // Get tile dimensions
        int tile_m = coord_space.tile.dim_m;
        int tile_n = coord_space.tile.dim_n;
        
        // Check bounds
        bool m_in_bounds = (thread_m >= 0 && thread_m < tile_m);
        bool n_in_bounds = thread_coord.is_full_dim_n || (thread_n >= 0 && thread_n < tile_n);
        
        return m_in_bounds && n_in_bounds;
    }
};

// Boundary-aware coordinate system - automatically manages boundary checks
template<typename ThreadCoordType>
struct boundary_aware_coord {
    ThreadCoordType coord;
    bool is_valid;
    
    template<typename CoordSpaceType, typename CoordType>
    __device__ __host__ constexpr boundary_aware_coord(CoordSpaceType coord_space, CoordType thread_coord) 
        : coord(thread_coord_map(coord_space, thread_coord)) {
        #ifdef __CUDA_ARCH__
        is_valid = boundary_extractor<ThreadCoordType>::check_bounds(coord_space, coord);
        #else
        is_valid = true; // Host compilation
        #endif
    }
    
    __device__ bool should_execute() const {
        return is_valid;
    }
    
    __device__ auto get_coord() const {
        return coord;
    }
};

// Smart boundary-aware thread coordinate mapping
template<typename CoordSpaceType, typename CoordType>
__device__ __host__ constexpr auto smart_thread_coord_map(CoordSpaceType coord_space, CoordType thread_coord) {
    using ThreadCoordType = thread_coord_t<CoordSpaceType>;
    return boundary_aware_coord<ThreadCoordType>(coord_space, thread_coord);
}

// Boundary condition generator - generates the actual condition for if statements
template<typename CoordSpaceType, typename CoordType>
struct boundary_condition_generator {
    CoordSpaceType space;
    CoordType thread_coord;
    
    __device__ __host__ constexpr boundary_condition_generator(CoordSpaceType s, CoordType tc) 
        : space(s), thread_coord(tc) {}
    
    // Get the condition string for debugging/code generation
    __device__ auto get_condition_check() const {
        return thread_coord.m < space.tile.dim_m && 
               (thread_coord.n < space.tile.dim_n || std::is_same_v<decltype(thread_coord.n), coord_full_dim_t>);
    }
    
    // More specific: get individual dimension bounds
    __device__ bool check_m_bound() const { return thread_coord.m < space.tile.dim_m; }
    __device__ bool check_n_bound() const { 
        return std::is_same_v<decltype(thread_coord.n), coord_full_dim_t> || thread_coord.n < space.tile.dim_n; 
    }
};

// Factory function for boundary condition generator
template<typename CoordSpaceType, typename CoordType>
__device__ __host__ constexpr auto make_boundary_condition(CoordSpaceType space, CoordType thread_coord) {
    return boundary_condition_generator<CoordSpaceType, CoordType>(space, thread_coord);
}

// ========================= Enhanced Memory Access Functions =========================

// Boundary-aware get_crd_ptr - automatically checks bounds before access
template<typename TensorType, int M, int N, typename ThreadCoordType>
__device__ auto get_crd_ptr_safe(TensorType (&tensor)[M][N], ThreadCoordType thread_coord) {
    // Runtime boundary check
    if (!boundary_extractor<ThreadCoordType>::check_bounds(thread_coord.base_coord, thread_coord)) {
        // Return nullptr or handle error gracefully
        return static_cast<TensorType*>(nullptr);
    }
    return get_crd_ptr(tensor, thread_coord);
}

// Boundary-aware register access
template<typename TensorType, int Size, typename ThreadCoordType>
__device__ auto get_crd_ptr_safe(TensorType (&reg_array)[Size], ThreadCoordType thread_coord) {
    auto& space = thread_coord.base_coord;
    int final_index = space.offset_m + thread_coord.thread_offset_m;
    
    if (final_index < 0 || final_index >= Size) {
        return static_cast<TensorType*>(nullptr);
    }
    return &reg_array[final_index];
}

// ========================= Conditional Execution Helpers =========================

// Macro for automatic boundary checking in coordinate operations
#define COORD_BOUNDARY_CHECK(coord_space, thread_coord) \
    (boundary_extractor<decltype(thread_coord_map(coord_space, thread_coord))>::check_bounds(coord_space, thread_coord_map(coord_space, thread_coord)))

// Smart execution wrapper - only execute if coordinates are valid
template<typename CoordSpaceType, typename CoordType, typename Func>
__device__ void execute_if_valid(CoordSpaceType coord_space, CoordType thread_coord, Func func) {
    auto mapped_coord = thread_coord_map(coord_space, thread_coord);
    if (boundary_extractor<decltype(mapped_coord)>::check_bounds(coord_space, mapped_coord)) {
        func(mapped_coord);
    }
}

// ========================= Compile-time Validation =========================

template<typename TileType>
struct is_crd_tile : std::false_type {};

template<int M, int N>
struct is_crd_tile<crd_tile_t<M, N>> : std::true_type {};

template<typename TileType>
constexpr bool is_crd_tile_v = is_crd_tile<TileType>::value;

// ========================= Advanced Coordinate Operations =========================

// Coordinate space composition for complex transformations
template<typename Space1, typename Space2>
__device__ __host__ constexpr auto compose_coord_spaces(Space1 space1, Space2 space2) {
    return coord_space_t<typename Space1::TileType>(
        space1.tile,
        space1.offset_m + space2.offset_m,
        space1.offset_n + space2.offset_n
    );
}

// Coordinate bounds checking (debug builds)
template<typename ThreadCoordType>
__device__ bool is_coord_in_bounds(ThreadCoordType thread_coord) {
    auto& space = thread_coord.base_coord;
    int final_m = space.offset_m + thread_coord.thread_offset_m;
    int final_n = space.offset_n + thread_coord.thread_offset_n;
    
    return (final_m >= 0 && final_m < space.tile.dim_m && 
            final_n >= 0 && final_n < space.tile.dim_n);
}

// ========================= Coordinate System Debugging =========================

#ifdef DEBUG_COORD_SYSTEM
template<typename ThreadCoordType>
__device__ void debug_print_coord(ThreadCoordType thread_coord, const char* name = "coord") {
    auto& space = thread_coord.base_coord;
    int final_m = space.offset_m + thread_coord.thread_offset_m;
    int final_n = space.offset_n + thread_coord.thread_offset_n;
    
    printf("%s: tile(%d,%d) offset(%d,%d) thread(%d,%d) final(%d,%d)\n",
           name, space.tile.dim_m, space.tile.dim_n,
           space.offset_m, space.offset_n,
           thread_coord.thread_offset_m, thread_coord.thread_offset_n,
           final_m, final_n);
}
#else
template<typename ThreadCoordType>
__device__ void debug_print_coord(ThreadCoordType thread_coord, const char* name = "coord") {}
#endif

// ========================= Coordinate System Optimizations =========================

// Compile-time coordinate calculation for static patterns
template<int BaseM, int BaseN, int ThreadM, int ThreadN, typename TileType>
struct static_coord_calculator {
    static constexpr int final_m = BaseM + ThreadM;
    static constexpr int final_n = BaseN + ThreadN;
    
    template<typename TensorType>
    __device__ static constexpr auto get_static_ptr(TensorType* tensor) {
        return &tensor[final_m * TileType::dim_n + final_n];
    }
};

// Coordinate vectorization helpers for aligned access patterns
template<typename ThreadCoordType, int VecSize>
__device__ bool is_vectorizable_coord(ThreadCoordType thread_coord) {
    auto& space = thread_coord.base_coord;
    int final_n = space.offset_n + thread_coord.thread_offset_n;
    return (final_n % VecSize == 0);
}

// ========================= Automatic Boundary Checking Integration =========================

// The coordinate system now automatically handles boundary checking through:
// 1. thread_coord_t automatically computes and stores boundary information
// 2. get_crd_ptr automatically checks bounds and returns safe dummy storage for invalid coordinates
// 3. All memory operations using coordinate-based pointers are automatically safe

// Summary of automatic boundary inference:
// - When thread_coord_map(space, coord(lane_id, _)) is called with CoordTile2D<MMA_M, MMA_N>
// - The system automatically infers that lane_id must be < MMA_M
// - get_crd_ptr returns safe dummy storage for threads with lane_id >= MMA_M
// - thread_copy operations are safe because they use the safe pointers from get_crd_ptr
// - No explicit boundary checks needed in user code!

// Debugging function to check coordinate validity
template<typename ThreadCoordType>
__device__ bool debug_coord_validity(ThreadCoordType thread_coord, const char* name = "coord") {
    bool valid = thread_coord.is_valid();
    #ifdef DEBUG_COORD_SYSTEM
    printf("%s: valid=%d, thread_m=%d, thread_n=%d, tile_m=%d, tile_n=%d\n", 
           name, valid, thread_coord.thread_offset_m, thread_coord.thread_offset_n,
           thread_coord.base_coord.tile.dim_m, thread_coord.base_coord.tile.dim_n);
    #endif
    return valid;
}