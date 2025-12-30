#pragma once

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cstdint>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

using namespace nvcuda;

// ========================= Type Aliases =========================
// Semantic integer types
using ShapeInt = int;        // for dynamic shape values like M, N, K
using TunableInt = int;      // for tunable parameters in kernel templates
using IndexInt = int;        // for loop indices and runtime calculated values
using TileInt = int;         // for computed tile dimensions like BM, BN, BK
using SmemAddr = uint32_t;   // for shared memory addresses in PTX instructions (ldmatrix, cp.async) - from __cvta_generic_to_shared or computed from them

// Memory space-aware pointer types for semantic clarity
template<typename T> using GlobalPtr = T*;      // Global memory pointers - for element-wise operations
template<typename T> using SharedPtr = T*;      // Shared memory pointers - for element-wise operations
template<typename T> using RegisterPtr = T*;    // Register array access (conceptual)

// Simplified Usage Guidelines:
// - Use SharedPtr<T>/GlobalPtr<T> for: ALL pointer operations and calculations 
// - Use SmemAddr ONLY for: shared memory addresses used in PTX instructions (ldmatrix, cp.async) - direct __cvta_generic_to_shared results or computed from them
// - Decision rule: Will this address be used in PTX instruction requiring shared memory address? → SmemAddr. Everything else → SharedPtr<T>/GlobalPtr<T>

// Predefined data types for clarity and portability
// 16-bit half-precision floating-point (FP16)
using fp16 = half;
using fp16_2 = half2;

// 16-bit bfloat16 floating-point (BF16)
using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

// 32-bit single-precision floating-point (FP32)
using fp32 = float;
using fp32_2 = float2;
using fp32_4 = float4;

// 64-bit double-precision floating-point (FP64)
using fp64 = double;
using fp64_2 = double2;

// Integer types
using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

// using ShapeInt = const int;
// using TunableInt = const int;
// using IndexInt = int;
// using TileInt = int;
// using PtrInt = uint32_t;

// ========================= Architecture Constants =========================
// Hardware architecture constants (use these directly instead of defining your own)
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_SHARED_MEM_PER_BLOCK 49152        // 48KB in bytes
#define CACHE_LINE_SIZE 128
#define MEMORY_BUS_WIDTH 128                   // bits

// Memory alignment constants (in bytes)
#define GLOBAL_MEM_ALIGN_BYTES 128             // 128-byte alignment for optimal global memory access
#define SHARED_MEM_BANK_SIZE 32                // 32 banks in shared memory
#define SHARED_MEM_BANK_WIDTH_BYTES 4          // 4 bytes per bank
#define VECTORIZED_ACCESS_WIDTH_BYTES 16       // 128-bit (16-byte) vectorized access

// NOTE: MMA register counts are implementation-specific and depend on:
// - Data types (half, float, etc.)
// - Register storage format (uint32_t fragments vs. native types)  
// - Specific MMA instruction variant
// Define these as needed in your specific kernel implementation.


// ========================= Utility Functions =========================
#define ceil_div(a, b) ((a + b - 1) / (b))

// ========================= Memory Swizzle Primitives =========================

// Swizzle permutation template for bank conflict avoidance
template<int kColStride = 16, int kStep = 8>
__device__ inline int swizzle_permuted_index(int i, int j) {
    static_assert(kColStride <= 16, "kColStride must <= 16");
    static_assert(kStep == 4 || kStep == 8, "kStep must be 8 or 4.");
    static_assert(kColStride % kStep == 0, "kColStride must be multiple of kStep.");
    
    if constexpr (kStep == 8) {
        return (((j >> 3) ^ (i >> 2)) % (kColStride >> 3)) << 3;
    } else {
        static_assert(kStep == 4);
        return (((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2;
    }
}

// Macro for backward compatibility
#define swizzle_permuted_index(i, j, col_stride, step) \
    swizzle_permuted_index<col_stride, step>(i, j)

// ========================= Memory Management Primitives =========================

// Memory allocation macros
#define mem_alloc_shared(dtype, ptr, shape) __shared__ dtype ptr shape
#define mem_alloc_shared_dynamic(dtype, ptr) extern __shared__ dtype ptr[]
#define mem_alloc_register(dtype, ptr, shape) dtype ptr shape

// Dynamic shared memory accessor - creates typed pointers from raw dynamic shared memory
template<typename T>
__device__ inline SharedPtr<T> shared_ptr_cast(void* raw_ptr) {
    return reinterpret_cast<SharedPtr<T>>(raw_ptr);
}

// Register pointer casting - creates typed pointers from register arrays
template<typename T, typename U>
__device__ inline RegisterPtr<T> register_ptr_cast(U* reg_ptr) {
    return reinterpret_cast<RegisterPtr<T>>(reg_ptr);
}

// Safe address calculation that avoids 64-bit type promotion
template<typename BaseAddr, typename Offset>
__device__ inline SmemAddr safe_addr_add(BaseAddr base, Offset offset) {
    return static_cast<SmemAddr>(base) + static_cast<uint32_t>(offset);
}

// Memory fill helper for recursive template implementation
template<typename T>
struct mem_fill_helper {
    // For non-array types (base case)
    template<typename U>
    __device__ static inline void fill_impl(T& elem, const U& value) {
        elem = static_cast<T>(value);
    }
};

// Specialization for array types
template<typename T, size_t N>
struct mem_fill_helper<T[N]> {
    // For array types, recursively fill each element
    template<typename U>
    __device__ static inline void fill_impl(T (&arr)[N], const U& value) {
        #pragma unroll
        for (size_t i = 0; i < N; ++i) {
            mem_fill_helper<T>::template fill_impl<U>(arr[i], value);
        }
    }
};

// Unified mem_fill interface that works with arbitrary dimensions
template<typename Array, typename ValueType>
__device__ inline void mem_fill(Array& arr, const ValueType& value) {
    mem_fill_helper<Array>::template fill_impl<ValueType>(arr, value);
}

// ========================= Synchronization Primitives =========================
#define block_sync() __syncthreads()

// ========================= Async Copy Primitives =========================

// Async copy cache policies
enum class AsyncCachePolicy {
    CA,  // Cache all levels (L1 + L2)
    CG   // Cache global level (L2 only)  
};

// CP_ASYNC instruction
#define CP_ASYNC_CA(dst, src, bytes) \
    asm volatile( \
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(bytes))

#define CP_ASYNC_CG(dst, src, bytes) \
    asm volatile( \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

// CP_ASYNC_BULK instruction macros (SM90+)
#define CP_ASYNC_BULK(dst, src, bytes) \
    asm volatile( \
        "cp.async.bulk.global.shared::cta.bulk_group.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(bytes))

#define CP_ASYNC_BULK_COMMIT_GROUP() asm volatile("cp.async.bulk.commit_group;\n" ::)
#define CP_ASYNC_BULK_WAIT_ALL() asm volatile("cp.async.bulk.wait_all;\n" ::)
#define CP_ASYNC_BULK_WAIT_GROUP(n) asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(n))

// Async copy dispatcher template
template<AsyncCachePolicy Policy>
struct async_copy_dispatcher {
    template<typename DstPtr, typename SrcPtr>
    __device__ static void execute(DstPtr dst_ptr, SrcPtr src_ptr, int bytes) {
        if constexpr (Policy == AsyncCachePolicy::CA) {
            CP_ASYNC_CA(dst_ptr, src_ptr, bytes);
        } else if constexpr (Policy == AsyncCachePolicy::CG) {
            CP_ASYNC_CG(dst_ptr, src_ptr, bytes);
        }
    }
};

// Async copy template functions
template<AsyncCachePolicy Policy, typename DstPtr, typename SrcPtr>
__device__ inline void async_copy_template(DstPtr dst_ptr, SrcPtr src_ptr, int bytes) {
    async_copy_dispatcher<Policy>::execute(dst_ptr, src_ptr, bytes);
}

// Async copy macro for backward compatibility
#define async_copy(dst_ptr, src_ptr, bytes, policy_str) \
    do { \
        if (__builtin_strcmp(policy_str, "ca") == 0) { \
            async_copy_template<AsyncCachePolicy::CA>(dst_ptr, src_ptr, bytes); \
        } else if (__builtin_strcmp(policy_str, "cg") == 0) { \
            async_copy_template<AsyncCachePolicy::CG>(dst_ptr, src_ptr, bytes); \
        } \
    } while(0)

// Async operation control macros
#define async_commit_group() CP_ASYNC_COMMIT_GROUP()
#define async_wait_group(n) CP_ASYNC_WAIT_GROUP(n)
#define async_wait_all() CP_ASYNC_WAIT_ALL()

// Async bulk operations macros  
#define async_bulk_copy(dst_ptr, src_ptr, bytes) CP_ASYNC_BULK(dst_ptr, src_ptr, bytes)
#define async_bulk_commit_group() CP_ASYNC_BULK_COMMIT_GROUP()
#define async_bulk_wait_group(n) CP_ASYNC_BULK_WAIT_GROUP(n)
#define async_bulk_wait_all() CP_ASYNC_BULK_WAIT_ALL()

// ========================= Copy Operation Primitives =========================

// Unified copy task types covering all memory space combinations
enum class CopyTask {
    G2S, // Global to Shared
    S2G, // Shared to Global
    G2R, // Global to Register
    R2G, // Register to Global
    S2R, // Shared to Register
    R2S  // Register to Shared
};

// Layout types
enum class Layout {
    ROW_MAJOR = 'n',
    COL_MAJOR = 't'
};

// Thread-level synchronous copy dispatcher - simple vectorized loads/stores
template<typename T, size_t BitWidth, CopyTask Task>
struct thread_copy_sync_dispatcher {
    template<typename SrcPtr, typename DstPtr>
    __device__ static void execute(SrcPtr src_ptr, DstPtr dst_ptr) {
        // Thread-level synchronous copies using vectorized loads/stores
        // Supported on all architectures (SM70+)
        if constexpr (BitWidth == 128) {
            (reinterpret_cast<float4 *>(dst_ptr)[0]) = (reinterpret_cast<float4 *>(src_ptr)[0]);
        } else if constexpr (BitWidth == 64) {
            (reinterpret_cast<float2 *>(dst_ptr)[0]) = (reinterpret_cast<float2 *>(src_ptr)[0]);
        } else if constexpr (BitWidth == 32) {
            (reinterpret_cast<half2 *>(dst_ptr)[0]) = (reinterpret_cast<half2 *>(src_ptr)[0]);
        } else if constexpr (BitWidth == 16) {
            (reinterpret_cast<half *>(dst_ptr)[0]) = (reinterpret_cast<half *>(src_ptr)[0]);
        } else {
            static_assert(BitWidth == 128 || BitWidth == 64 || BitWidth == 32 || BitWidth == 16, 
                          "Unsupported BitWidth for thread_copy_sync");
        }
    }
};

// Thread-level synchronous copy API - supports all memory space combinations
template<typename T, size_t BitWidth, CopyTask Task, typename SrcPtr, typename DstPtr>
__device__ inline void thread_copy_sync(SrcPtr src_ptr, DstPtr dst_ptr) {
    // Supports all CopyTask combinations: G2S, S2G, G2R, R2G, S2R, R2S
    // Available on all architectures (SM70+)
    static_assert(Task == CopyTask::G2S || Task == CopyTask::S2G || 
                  Task == CopyTask::G2R || Task == CopyTask::R2G ||
                  Task == CopyTask::S2R || Task == CopyTask::R2S,
                  "thread_copy_sync supports all memory space combinations");
    thread_copy_sync_dispatcher<T, BitWidth, Task>::execute(src_ptr, dst_ptr);
}

// ========================= Warp-Level Async Copy Primitives =========================

// CP.ASYNC instruction template dispatcher (SM80+ Ampere/Ada/Hopper)
template<typename T, size_t BitWidth, CopyTask Task>
struct warp_copy_async_dispatcher {
    template<typename SrcPtr, typename DstPtr>
    __device__ static void execute(SrcPtr src_ptr, DstPtr dst_ptr) {
        constexpr size_t bytes = BitWidth / 8;
        
        static_assert(Task == CopyTask::G2S, 
                      "CP.ASYNC only supports G2S transfers (Global to Shared)");
        
        // Check architecture support
#if __CUDA_ARCH__ >= 800  // SM80+ (Ampere, Ada, Hopper)
        // Global to Shared: src is global pointer, dst is SmemAddr
        SmemAddr dst_addr = static_cast<SmemAddr>(dst_ptr);
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" 
            :: "r"(dst_addr), "l"(src_ptr), "n"(bytes));
#else
        // static_assert(false, "CP.ASYNC requires SM80+ (Ampere/Ada/Hopper). Use thread_copy_sync for older architectures.");
        // 有 static_assert(false, ...) - 这在模板实例化时会无条件失败，即使代码永远不会执行到那个分支。
        // CP.ASYNC not available, fallback to regular copy
        thread_copy_sync_dispatcher<T, BitWidth, Task>::execute(src_ptr, dst_ptr);
#endif
    }
};

// Warp-level asynchronous copy operations (CP.ASYNC - SM80+)
template<typename T, size_t BitWidth, CopyTask Task, typename SrcPtr, typename DstPtr>
__device__ inline void warp_copy_async(SrcPtr src_ptr, DstPtr dst_ptr) {
    // Only supports CopyTask::G2S (Global to Shared)
    // Requires SM80+ (Ampere/Ada/Hopper architectures)
    static_assert(Task == CopyTask::G2S, 
                  "warp_copy_async only supports G2S transfers (Global to Shared). Use thread_copy_sync for other directions.");
    warp_copy_async_dispatcher<T, BitWidth, Task>::execute(src_ptr, dst_ptr);
}

// Warp-level async copy control functions
__device__ inline void warp_copy_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ inline void warp_copy_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ========================= Warp-Level Copy Primitives =========================

// PTX LDMATRIX instruction macros
#define LDMATRIX_X1(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" \
                 : "=r"(R) : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                 : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr) \
    asm volatile( \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) \
        : "r"(addr))

#define LDMATRIX_X1_T(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" \
                 : "=r"(R) : "r"(addr))

#define LDMATRIX_X2_T(R0, R1, addr) \
    asm volatile( \
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" \
        : "=r"(R0), "=r"(R1) \
        : "r"(addr))

#define LDMATRIX_X4_T(R0, R1, R2, R3, addr) \
    asm volatile( \
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) \
        : "r"(addr))

// PTX STMATRIX instruction macros (SM90+)
#define STMATRIX_X1(addr, R) \
    asm volatile( \
        "stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" ::"r"(addr), "r"(R))

#define STMATRIX_X2(addr, R0, R1) \
    asm volatile( \
        "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(addr), \
        "r"(R0), "r"(R1))

#define STMATRIX_X4(addr, R0, R1, R2, R3) \
    asm volatile( \
        "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), \
        "r"(R0), "r"(R1), "r"(R2), "r"(R3))

#define STMATRIX_X1_T(addr, R) \
    asm volatile( \
        "stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n" ::"r"(addr), "r"(R))

#define STMATRIX_X2_T(addr, R0, R1) \
    asm volatile( \
        "stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(addr), \
        "r"(R0), "r"(R1))

#define STMATRIX_X4_T(addr, R0, R1, R2, R3) \
    asm volatile( \
        "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), \
        "r"(R0), "r"(R1), "r"(R2), "r"(R3))

// Template-based parameter extraction for variadic arguments
template<size_t N, typename... Args>
struct get_nth_arg;

template<typename T, typename... Args>
struct get_nth_arg<0, T, Args...> {
    using type = T;
    __device__ __host__ static constexpr T& get(T& first, Args&... args) { return first; }
};

template<size_t N, typename T, typename... Args>
struct get_nth_arg<N, T, Args...> {
    using type = typename get_nth_arg<N-1, Args...>::type;
    __device__ __host__ static constexpr auto& get(T& first, Args&... args) {
        return get_nth_arg<N-1, Args...>::get(args...);
    }
};

template<size_t N, typename... Args>
__device__ __host__ constexpr auto& get_arg(Args&... args) {
    return get_nth_arg<N, Args...>::get(args...);
}

// LDMATRIX/STMATRIX instruction template dispatcher
template<typename T, size_t BitWidth, CopyTask Task, Layout LayoutType>
struct warp_sync_copy_dispatcher {
    template<typename... Args>
    __device__ static void execute(SmemAddr smem_addr, Args&... args) {
        if constexpr (Task == CopyTask::S2R) {
            // LDMATRIX: Shared to Register (SM80+ Ampere/Ada/Hopper)
#if __CUDA_ARCH__ >= 800  // SM80+
            if constexpr (LayoutType == Layout::ROW_MAJOR) {
                if constexpr (BitWidth == 32 && sizeof...(args) == 1) {
                    LDMATRIX_X1(get_arg<0>(args...), smem_addr);
                } else if constexpr (BitWidth == 64 && sizeof...(args) == 2) {
                    LDMATRIX_X2(get_arg<0>(args...), get_arg<1>(args...), smem_addr);
                } else if constexpr (BitWidth == 128 && sizeof...(args) == 4) {
                    LDMATRIX_X4(get_arg<0>(args...), get_arg<1>(args...), get_arg<2>(args...), get_arg<3>(args...), smem_addr);
                }
            } else if constexpr (LayoutType == Layout::COL_MAJOR) {
                if constexpr (BitWidth == 32 && sizeof...(args) == 1) {
                    LDMATRIX_X1_T(get_arg<0>(args...), smem_addr);
                } else if constexpr (BitWidth == 64 && sizeof...(args) == 2) {
                    LDMATRIX_X2_T(get_arg<0>(args...), get_arg<1>(args...), smem_addr);
                } else if constexpr (BitWidth == 128 && sizeof...(args) == 4) {
                    LDMATRIX_X4_T(get_arg<0>(args...), get_arg<1>(args...), get_arg<2>(args...), get_arg<3>(args...), smem_addr);
                }
            }
#else
            // static_assert(false, "LDMATRIX requires SM80+ (Ampere/Ada/Hopper). Use thread_copy_sync for older architectures.");
            // LDMATRIX not available, fallback to regular copy
            // LDMATRIX/STMATRIX fallback not implemented - requires manual register handling
            // TODO
#endif
        } else if constexpr (Task == CopyTask::R2S) {
            // STMATRIX: Register to Shared (SM90+ Hopper only)
#if __CUDA_ARCH__ >= 900  // SM90+ Hopper
            if constexpr (LayoutType == Layout::ROW_MAJOR) {
                if constexpr (BitWidth == 32 && sizeof...(args) == 1) {
                    STMATRIX_X1(smem_addr, get_arg<0>(args...));
                } else if constexpr (BitWidth == 64 && sizeof...(args) == 2) {
                    STMATRIX_X2(smem_addr, get_arg<0>(args...), get_arg<1>(args...));
                } else if constexpr (BitWidth == 128 && sizeof...(args) == 4) {
                    STMATRIX_X4(smem_addr, get_arg<0>(args...), get_arg<1>(args...), get_arg<2>(args...), get_arg<3>(args...));
                }
            } else if constexpr (LayoutType == Layout::COL_MAJOR) {
                if constexpr (BitWidth == 32 && sizeof...(args) == 1) {
                    STMATRIX_X1_T(smem_addr, get_arg<0>(args...));
                } else if constexpr (BitWidth == 64 && sizeof...(args) == 2) {
                    STMATRIX_X2_T(smem_addr, get_arg<0>(args...), get_arg<1>(args...));
                } else if constexpr (BitWidth == 128 && sizeof...(args) == 4) {
                    STMATRIX_X4_T(smem_addr, get_arg<0>(args...), get_arg<1>(args...), get_arg<2>(args...), get_arg<3>(args...));
                }
            }
#else
            // static_assert(false, "STMATRIX requires SM90+ (Hopper). Use thread_copy_sync for Ampere/Ada architectures.");
            // STMATRIX not available, fallback to regular copy
            // LDMATRIX/STMATRIX fallback not implemented - requires manual register handling
            // TODO
#endif
        }
    }
};

// Warp-level synchronous copy API - LDMATRIX/STMATRIX operations
template<typename T, size_t BitWidth, CopyTask Task, Layout LayoutType, typename... Args>
__device__ inline void warp_copy_sync(SmemAddr smem_addr, Args&... args) {
    // LDMATRIX (S2R): SM80+ Ampere/Ada/Hopper
    // STMATRIX (R2S): SM90+ Hopper only
    static_assert(Task == CopyTask::S2R || Task == CopyTask::R2S, 
                  "warp_copy_sync only supports S2R (ldmatrix) and R2S (stmatrix) transfers");
    warp_sync_copy_dispatcher<T, BitWidth, Task, LayoutType>::execute(smem_addr, args...);
}

// ========================= MMA Compute Primitives =========================

// MMA shape and layout enums
enum class MmaShape {
    M16N8K16,
    M32N8K16  // extensible for future shapes
};

enum class MmaLayout {
    TN, // Transposed A, Non-transposed B (row.col)
    NT, // Non-transposed A, Transposed B (col.row)
    NN, // Non-transposed A, Non-transposed B
    TT  // Transposed A, Transposed B
};

// PTX MMA instruction macro
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) \
    asm volatile( \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
        : "=r"(RD0), "=r"(RD1) \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

// MMA dispatcher templates
template<MmaShape Shape, MmaLayout LayoutType, typename ABType, typename CDType>
struct mma_dispatcher {
    template<typename... Args>
    __device__ static void execute(Args&... args) {
        static_assert(sizeof...(args) == 10, "MMA m16n8k16 requires exactly 10 register arguments");
    }
};

// Specialization for M16N8K16 with half precision TN layout (row.col)
template<>
struct mma_dispatcher<MmaShape::M16N8K16, MmaLayout::TN, __half, __half> {
    template<typename... Args>
    __device__ static void execute(Args&... args) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
            " {%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
            : "=r"(get_arg<0>(args...)), "=r"(get_arg<1>(args...))
            : "r"(get_arg<2>(args...)), "r"(get_arg<3>(args...)), "r"(get_arg<4>(args...)), "r"(get_arg<5>(args...)),
              "r"(get_arg<6>(args...)), "r"(get_arg<7>(args...)),
              "r"(get_arg<8>(args...)), "r"(get_arg<9>(args...))
        );
    }
};

// Specialization for M16N8K16 with half precision NT layout (col.row)  
template<>
struct mma_dispatcher<MmaShape::M16N8K16, MmaLayout::NT, __half, __half> {
    template<typename... Args>
    __device__ static void execute(Args&... args) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.col.row.f16.f16.f16.f16"
            " {%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
            : "=r"(get_arg<0>(args...)), "=r"(get_arg<1>(args...))
            : "r"(get_arg<2>(args...)), "r"(get_arg<3>(args...)), "r"(get_arg<4>(args...)), "r"(get_arg<5>(args...)),
              "r"(get_arg<6>(args...)), "r"(get_arg<7>(args...)),
              "r"(get_arg<8>(args...)), "r"(get_arg<9>(args...))
        );
    }
};

// Clean template-based MMA API - all parameters as template arguments
template<MmaShape Shape, MmaLayout LayoutType, typename ABType, typename CDType, typename... Args>
__device__ inline void mma(Args&... args) {
    mma_dispatcher<Shape, LayoutType, ABType, CDType>::execute(args...);
}

// ========================= Control Flow Primitives =========================

// Serial loop template for range-based operations
template<typename IndexType, IndexType Start, IndexType End, IndexType Step, typename Func>
__device__ inline void serial_range_for_template(Func&& func) {
    #pragma unroll
    for (IndexType i = Start; i < End; i += Step) {
        func(i);
    }
}

// Serial loop macro for range-based operations
#define serial_range_for(iterator, start, end, step) \
    _Pragma("unroll") \
    for (IndexInt iterator = start; iterator < end; iterator += step)

// ========================= Warp Shuffle Primitives =========================

// Basic warp shuffle operations - support both full-warp and sub-warp modes
template<typename T>
__device__ inline T warp_shuffle(T value, int src_lane, unsigned mask = 0xffffffff) {
    return __shfl_sync(mask, value, src_lane);
}

template<typename T>
__device__ inline T warp_shuffle_width(T value, int src_lane, int width, unsigned mask = 0xffffffff) {
    return __shfl_sync(mask, value, src_lane, width);
}

// Warp shuffle with offset - full warp operations
template<typename T>  
__device__ inline T warp_shuffle_down(T value, int offset, unsigned mask = 0xffffffff) {
    return __shfl_down_sync(mask, value, offset);
}

template<typename T>
__device__ inline T warp_shuffle_up(T value, int offset, unsigned mask = 0xffffffff) {
    return __shfl_up_sync(mask, value, offset);
}

template<typename T>
__device__ inline T warp_shuffle_xor(T value, int mask_val, unsigned mask = 0xffffffff) {
    return __shfl_xor_sync(mask, value, mask_val);
}

// Sub-warp shuffle operations with explicit width parameter
template<typename T>
__device__ inline T warp_shuffle_xor_width(T value, int mask_val, int width, unsigned mask = 0xffffffff) {
    return __shfl_xor_sync(mask, value, mask_val, width);
}

// ========================= Advanced Warp Shuffle Patterns =========================

// MMA result redistribution - spread single value to consecutive lanes
template<typename T>
__device__ inline void warp_shuffle_spread_x4(T value, T output[4], IndexInt lane_id, int width = 32, unsigned mask = 0xffffffff) {
    output[0] = value;                                        // Current lane
    output[1] = __shfl_sync(mask, value, lane_id + 1, width); // Next lane
    output[2] = __shfl_sync(mask, value, lane_id + 2, width); // Lane + 2  
    output[3] = __shfl_sync(mask, value, lane_id + 3, width); // Lane + 3
}

// Sub-warp version for explicit width specification (common in Flash Attention)
template<typename T>
__device__ inline void warp_shuffle_spread_x4_width(T value, T output[4], IndexInt lane_id, int width, unsigned mask = 0xffffffff) {
    output[0] = value;
    output[1] = __shfl_sync(mask, value, lane_id + 1, width);
    output[2] = __shfl_sync(mask, value, lane_id + 2, width);
    output[3] = __shfl_sync(mask, value, lane_id + 3, width);
}

// MMA result redistribution for dual accumulator pattern (common in GEMM)
template<typename T>
__device__ inline void warp_shuffle_spread_dual_x4(T src0, T src1, T dst0[4], T dst1[4], IndexInt lane_id, int width = 32, unsigned mask = 0xffffffff) {
    warp_shuffle_spread_x4(src0, dst0, lane_id, width, mask);
    warp_shuffle_spread_x4(src1, dst1, lane_id, width, mask);
}

// Sub-warp version for explicit width specification
template<typename T>
__device__ inline void warp_shuffle_spread_dual_x4_width(T src0, T src1, T dst0[4], T dst1[4], IndexInt lane_id, int width, unsigned mask = 0xffffffff) {
    warp_shuffle_spread_x4_width(src0, dst0, lane_id, width, mask);
    warp_shuffle_spread_x4_width(src1, dst1, lane_id, width, mask);
}

// Broadcast value from specific lane to all lanes in warp
template<typename T>
__device__ inline T warp_shuffle_broadcast(T value, IndexInt src_lane, unsigned mask = 0xffffffff) {
    return __shfl_sync(mask, value, src_lane);
}

// ========================= Warp Reduction Primitives =========================

// Generic warp reduction using butterfly shuffle (XOR-based) - full warp
template<typename T, typename BinaryOp>
__device__ inline T warp_shuffle_reduce(T value, BinaryOp op, unsigned mask = 0xffffffff) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_xor_sync(mask, value, offset);
        value = op(value, other);
    }
    return value;
}

// Generic sub-warp reduction using butterfly shuffle with explicit width
template<typename T, typename BinaryOp>
__device__ inline T warp_shuffle_reduce_width(T value, BinaryOp op, int width, unsigned mask = 0xffffffff) {
    #pragma unroll
    for (int offset = width / 2; offset > 0; offset /= 2) {
        T other = __shfl_xor_sync(mask, value, offset, width);
        value = op(value, other);
    }
    return value;
}

// Specialized reductions for different data types - full warp
template<typename T>
__device__ inline T warp_shuffle_sum(T value, unsigned mask = 0xffffffff);

// FP32 sum reduction - full warp
template<>
__device__ inline fp32 warp_shuffle_sum<fp32>(fp32 value, unsigned mask) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(mask, value, offset);
    }
    return value;
}

// Sub-warp sum reductions with explicit width
template<typename T>
__device__ inline T warp_shuffle_sum_width(T value, int width, unsigned mask = 0xffffffff);

// FP32 sum reduction - sub-warp with width
template<>
__device__ inline fp32 warp_shuffle_sum_width<fp32>(fp32 value, int width, unsigned mask) {
    #pragma unroll
    for (int offset = width / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(mask, value, offset, width);
    }
    return value;
}

// FP16 sum reduction - full warp
template<>
__device__ inline fp16 warp_shuffle_sum<fp16>(fp16 value, unsigned mask) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value = __hadd(value, __shfl_xor_sync(mask, value, offset));
    }
    return value;
}

// FP16 sum reduction - sub-warp with width  
template<>
__device__ inline fp16 warp_shuffle_sum_width<fp16>(fp16 value, int width, unsigned mask) {
    #pragma unroll
    for (int offset = width / 2; offset > 0; offset /= 2) {
        value = __hadd(value, __shfl_xor_sync(mask, value, offset, width));
    }
    return value;
}

// BF16 sum reduction
template<>
__device__ inline bf16 warp_shuffle_sum<bf16>(bf16 value, unsigned mask) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value = __hadd(value, __shfl_xor_sync(mask, value, offset));
    }
    return value;
}

// FP16 to FP32 sum reduction (higher precision accumulation)
__device__ inline fp32 warp_shuffle_sum_f16_to_f32(fp16 value, unsigned mask = 0xffffffff) {
    fp32 value_f32 = __half2float(value);
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value_f32 += __shfl_xor_sync(mask, value_f32, offset);
    }
    return value_f32;
}

// BF16 to FP32 sum reduction
__device__ inline fp32 warp_shuffle_sum_bf16_to_f32(bf16 value, unsigned mask = 0xffffffff) {
    fp32 value_f32 = __bfloat162float(value);
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value_f32 += __shfl_xor_sync(mask, value_f32, offset);
    }
    return value_f32;
}

// INT8 to INT32 sum reduction
__device__ inline IndexInt warp_shuffle_sum_i8_to_i32(int8_t value, unsigned mask = 0xffffffff) {
    IndexInt value_i32 = static_cast<IndexInt>(value);
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value_i32 += __shfl_xor_sync(mask, value_i32, offset);
    }
    return value_i32;
}

// INT32 sum reduction
__device__ inline IndexInt warp_shuffle_sum_i32(IndexInt value, unsigned mask = 0xffffffff) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(mask, value, offset);
    }
    return value;
}

// FP8 E4M3 to FP16 sum reduction
__device__ inline fp16 warp_shuffle_sum_fp8_e4m3_to_f16(__nv_fp8_storage_t value, unsigned mask = 0xffffffff) {
    fp16 value_f16 = __nv_cvt_fp8_to_halfraw(value, __NV_E4M3);
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value_f16 = __hadd(value_f16, __shfl_xor_sync(mask, value_f16, offset));
    }
    return value_f16;
}

// FP8 E5M2 to FP16 sum reduction
__device__ inline fp16 warp_shuffle_sum_fp8_e5m2_to_f16(__nv_fp8_storage_t value, unsigned mask = 0xffffffff) {
    fp16 value_f16 = __nv_cvt_fp8_to_halfraw(value, __NV_E5M2);
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value_f16 = __hadd(value_f16, __shfl_xor_sync(mask, value_f16, offset));
    }
    return value_f16;
}

// Generic max reduction - full warp
template<typename T>
__device__ inline T warp_shuffle_max(T value, unsigned mask = 0xffffffff) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_xor_sync(mask, value, offset);
        value = value > other ? value : other;
    }
    return value;
}

// Generic max reduction - sub-warp with width
template<typename T>
__device__ inline T warp_shuffle_max_width(T value, int width, unsigned mask = 0xffffffff) {
    #pragma unroll
    for (int offset = width / 2; offset > 0; offset /= 2) {
        T other = __shfl_xor_sync(mask, value, offset, width);
        value = value > other ? value : other;
    }
    return value;
}

// FP32 max reduction (using fmaxf) - full warp
template<>
__device__ inline fp32 warp_shuffle_max<fp32>(fp32 value, unsigned mask) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(mask, value, offset));
    }
    return value;
}

// FP32 max reduction (using fmaxf) - sub-warp with width
template<>
__device__ inline fp32 warp_shuffle_max_width<fp32>(fp32 value, int width, unsigned mask) {
    #pragma unroll
    for (int offset = width / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(mask, value, offset, width));
    }
    return value;
}

// Generic min reduction - full warp
template<typename T>
__device__ inline T warp_shuffle_min(T value, unsigned mask = 0xffffffff) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_xor_sync(mask, value, offset);
        value = value < other ? value : other;
    }
    return value;
}

// Generic min reduction - sub-warp with width
template<typename T>
__device__ inline T warp_shuffle_min_width(T value, int width, unsigned mask = 0xffffffff) {
    #pragma unroll
    for (int offset = width / 2; offset > 0; offset /= 2) {
        T other = __shfl_xor_sync(mask, value, offset, width);
        value = value < other ? value : other;
    }
    return value;
}

// FP32 min reduction (using fminf) - full warp
template<>
__device__ inline fp32 warp_shuffle_min<fp32>(fp32 value, unsigned mask) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value = fminf(value, __shfl_xor_sync(mask, value, offset));
    }
    return value;
}

// FP32 min reduction (using fminf) - sub-warp with width
template<>
__device__ inline fp32 warp_shuffle_min_width<fp32>(fp32 value, int width, unsigned mask) {
    #pragma unroll
    for (int offset = width / 2; offset > 0; offset /= 2) {
        value = fminf(value, __shfl_xor_sync(mask, value, offset, width));
    }
    return value;
}

// Warp-level prefix scan (exclusive)
template<typename T, typename BinaryOp>
__device__ inline T warp_shuffle_scan_exclusive(T value, BinaryOp op, unsigned mask = 0xffffffff) {
    IndexInt lane_id = threadIdx.x % WARP_SIZE;
    T result = T{};  // Identity for exclusive scan
    
    #pragma unroll
    for (IndexInt offset = 1; offset < WARP_SIZE; offset *= 2) {
        T temp = __shfl_up_sync(mask, value, offset);
        if (lane_id >= offset) {
            value = op(value, temp);
        }
    }
    
    // Shift right for exclusive scan
    result = __shfl_up_sync(mask, value, 1);
    if (lane_id == 0) result = T{};
    
    return result;
}

// Warp-level prefix sum (exclusive) - common case
template<typename T>
__device__ inline T warp_shuffle_prefix_sum(T value, unsigned mask = 0xffffffff) {
    return warp_shuffle_scan_exclusive(value, [](T a, T b) { return a + b; }, mask);
}

// Rotate values within warp (circular shift)
template<typename T>
__device__ inline T warp_shuffle_rotate(T value, IndexInt offset, unsigned mask = 0xffffffff) {
    IndexInt lane_id = threadIdx.x % WARP_SIZE;
    IndexInt src_lane = (lane_id + offset) % WARP_SIZE;
    return __shfl_sync(mask, value, src_lane);
}

// Gather values from multiple lanes
template<typename T, IndexInt N>
__device__ inline void warp_shuffle_gather(T src_value, IndexInt src_lanes[N], T dst_values[N], unsigned mask = 0xffffffff) {
    #pragma unroll
    for (IndexInt i = 0; i < N; ++i) {
        dst_values[i] = __shfl_sync(mask, src_value, src_lanes[i]);
    }
}

// Matrix transpose pattern for 2x2 blocks using XOR shuffle
template<typename T>
__device__ inline void warp_shuffle_transpose_2x2(T values[4], unsigned mask = 0xffffffff) {
    IndexInt lane_id = threadIdx.x % WARP_SIZE;
    
    // Transpose 2x2 blocks using XOR pattern
    T temp0 = __shfl_xor_sync(mask, values[0], 1);
    T temp1 = __shfl_xor_sync(mask, values[1], 1);
    T temp2 = __shfl_xor_sync(mask, values[2], 1);
    T temp3 = __shfl_xor_sync(mask, values[3], 1);
    
    if (lane_id & 1) {
        values[0] = temp1;
        values[1] = temp0;
        values[2] = temp3; 
        values[3] = temp2;
    } else {
        values[1] = temp1;
        values[3] = temp3;
    }
}

// ========================= Struct Shuffle Support =========================

// Shuffle for 2-element structs (e.g., {m, d} in softmax)
template<typename T1, typename T2>
struct pair_struct {
    T1 first;
    T2 second;
};

template<typename T1, typename T2>
__device__ inline pair_struct<T1, T2> warp_shuffle_struct(pair_struct<T1, T2> value, IndexInt src_lane, unsigned mask = 0xffffffff) {
    pair_struct<T1, T2> result;
    result.first = __shfl_sync(mask, value.first, src_lane);
    result.second = __shfl_sync(mask, value.second, src_lane);
    return result;
}

template<typename T1, typename T2>
__device__ inline pair_struct<T1, T2> warp_shuffle_struct_xor(pair_struct<T1, T2> value, IndexInt offset, unsigned mask = 0xffffffff) {
    pair_struct<T1, T2> result;
    result.first = __shfl_xor_sync(mask, value.first, offset);
    result.second = __shfl_xor_sync(mask, value.second, offset);
    return result;
}

// Reduce pair structs with custom operation
template<typename T1, typename T2, typename BinaryOp>
__device__ inline pair_struct<T1, T2> warp_shuffle_reduce_struct(pair_struct<T1, T2> value, BinaryOp op, unsigned mask = 0xffffffff) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        pair_struct<T1, T2> other = warp_shuffle_struct_xor(value, offset, mask);
        value = op(value, other);
    }
    return value;
}

// ========================= Block Reduction Primitives =========================

// Block-level sum reduction with shared memory
template<typename T, IndexInt NUM_THREADS>
__device__ inline T block_reduce_sum(T thread_value) {
    constexpr IndexInt NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ T reduce_smem[NUM_WARPS];
    
    IndexInt tid = threadIdx.x;
    IndexInt warp_id = tid / WARP_SIZE;
    IndexInt lane_id = tid % WARP_SIZE;
    
    // Warp-level reduction
    T warp_sum = warp_shuffle_sum(thread_value);
    
    // Store warp result to shared memory
    if (lane_id == 0) {
        reduce_smem[warp_id] = warp_sum;
    }
    block_sync();
    
    // Final reduction among warp leaders
    T block_sum = (lane_id < NUM_WARPS) ? reduce_smem[lane_id] : T{};
    if (warp_id == 0) {
        block_sum = warp_shuffle_sum(block_sum);
    }
    
    return block_sum;
}

// Block-level max reduction
template<typename T, IndexInt NUM_THREADS>
__device__ inline T block_reduce_max(T thread_value) {
    constexpr IndexInt NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ T reduce_smem[NUM_WARPS];
    
    IndexInt tid = threadIdx.x;
    IndexInt warp_id = tid / WARP_SIZE;
    IndexInt lane_id = tid % WARP_SIZE;
    
    // Warp-level reduction
    T warp_max = warp_shuffle_max(thread_value);
    
    // Store warp result to shared memory
    if (lane_id == 0) {
        reduce_smem[warp_id] = warp_max;
    }
    block_sync();
    
    // Final reduction among warp leaders
    T block_max = (lane_id < NUM_WARPS) ? reduce_smem[lane_id] : thread_value; // Use original value as default
    if (warp_id == 0) {
        block_max = warp_shuffle_max(block_max);
    }
    
    return block_max;
}

// Block-level reduction with type conversion (e.g., FP16 accumulate to FP32)
template<typename InputT, typename AccumT, IndexInt NUM_THREADS>
__device__ inline AccumT block_reduce_sum_with_cast(InputT thread_value) {
    constexpr IndexInt NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ AccumT reduce_smem[NUM_WARPS];
    
    IndexInt tid = threadIdx.x;
    IndexInt warp_id = tid / WARP_SIZE;
    IndexInt lane_id = tid % WARP_SIZE;
    
    // Convert and perform warp-level reduction
    AccumT converted_value = static_cast<AccumT>(thread_value);
    AccumT warp_sum = warp_shuffle_sum(converted_value);
    
    // Store warp result to shared memory
    if (lane_id == 0) {
        reduce_smem[warp_id] = warp_sum;
    }
    block_sync();
    
    // Final reduction among warp leaders
    AccumT block_sum = (lane_id < NUM_WARPS) ? reduce_smem[lane_id] : AccumT{};
    if (warp_id == 0) {
        block_sum = warp_shuffle_sum(block_sum);
    }
    
    return block_sum;
}

// Specialized block reduction for FP16 to FP32
template<IndexInt NUM_THREADS>
__device__ inline fp32 block_reduce_sum_f16_to_f32(fp16 thread_value) {
    constexpr IndexInt NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ fp32 reduce_smem[NUM_WARPS];
    
    IndexInt tid = threadIdx.x;
    IndexInt warp_id = tid / WARP_SIZE;
    IndexInt lane_id = tid % WARP_SIZE;
    
    // Warp-level reduction with type conversion
    fp32 warp_sum = warp_shuffle_sum_f16_to_f32(thread_value);
    
    if (lane_id == 0) {
        reduce_smem[warp_id] = warp_sum;
    }
    block_sync();
    
    fp32 block_sum = (lane_id < NUM_WARPS) ? reduce_smem[lane_id] : 0.0f;
    if (warp_id == 0) {
        block_sum = warp_shuffle_sum(block_sum);
    }
    
    return block_sum;
}

// Specialized block reduction for BF16 to FP32
template<IndexInt NUM_THREADS>
__device__ inline fp32 block_reduce_sum_bf16_to_f32(bf16 thread_value) {
    constexpr IndexInt NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ fp32 reduce_smem[NUM_WARPS];
    
    IndexInt tid = threadIdx.x;
    IndexInt warp_id = tid / WARP_SIZE;
    IndexInt lane_id = tid % WARP_SIZE;
    
    // Warp-level reduction with type conversion
    fp32 warp_sum = warp_shuffle_sum_bf16_to_f32(thread_value);
    
    if (lane_id == 0) {
        reduce_smem[warp_id] = warp_sum;
    }
    block_sync();
    
    fp32 block_sum = (lane_id < NUM_WARPS) ? reduce_smem[lane_id] : 0.0f;
    if (warp_id == 0) {
        block_sum = warp_shuffle_sum(block_sum);
    }
    
    return block_sum;
}



// ========================= Address Conversion Primitives =========================

// Address space conversion utilities for copy instructions
__device__ inline SmemAddr generic_to_shared_addr(void* ptr) {
    return __cvta_generic_to_shared(ptr);
}

// ========================= Arithmetic Compute Primitives =========================

// Basic arithmetic operations (extensible)
template<typename T>
__device__ inline T add(const T& a, const T& b) {
    return a + b;
}

template<typename T>
__device__ inline T sub(const T& a, const T& b) {
    return a - b;
}

template<typename T>
__device__ inline T mul(const T& a, const T& b) {
    return a * b;
}

template<typename T>
__device__ inline T fma(const T& a, const T& b, const T& c) {
    return a * b + c;
}

// ========================= High-Precision Math Functions =========================

// Exponential functions
template<typename T>
__device__ inline T compute_exp(const T& x);

// FP32 exponential with high precision
template<>
__device__ inline fp32 compute_exp<fp32>(const fp32& x) {
    return __expf(x);
}

// Reciprocal functions
template<typename T>
__device__ inline T compute_rcp(const T& x);

// FP32 reciprocal with round-to-nearest
template<>
__device__ inline fp32 compute_rcp<fp32>(const fp32& x) {
    return __frcp_rn(x);
}

// Maximum functions
template<typename T>
__device__ inline T compute_max(const T& a, const T& b);

// FP32 maximum
template<>
__device__ inline fp32 compute_max<fp32>(const fp32& a, const fp32& b) {
    return fmaxf(a, b);
}

// FP16 maximum  
template<>
__device__ inline fp16 compute_max<fp16>(const fp16& a, const fp16& b) {
    return __hmax(a, b);
}

// Generic maximum (for other types)
template<typename T>
__device__ inline T compute_max(const T& a, const T& b) {
    return a > b ? a : b;
}

// Fused multiply-add functions
template<typename T>
__device__ inline T compute_fma(const T& a, const T& b, const T& c);

// FP32 fused multiply-add with round-to-nearest
template<>
__device__ inline fp32 compute_fma<fp32>(const fp32& a, const fp32& b, const fp32& c) {
    return __fmaf_rn(a, b, c);
}

// ========================= Type Conversion Primitives =========================

// High-precision type conversions
template<typename SrcT, typename DstT>
__device__ inline DstT compute_cast(const SrcT& value);

// FP16 to FP32 conversion
template<>
__device__ inline fp32 compute_cast<fp16, fp32>(const fp16& value) {
    return __half2float(value);
}

// FP32 to FP16 conversion with round-to-nearest
template<>
__device__ inline fp16 compute_cast<fp32, fp16>(const fp32& value) {
    return __float2half_rn(value);
}

// ========================= Register Data Access Primitives =========================

// Safe register reinterpretation with semantic clarity
template<typename DstT, typename SrcT>
__device__ inline DstT* register_cast(SrcT* reg_ptr) {
    return reinterpret_cast<DstT*>(reg_ptr);
}

// Convenience aliases for common register casting patterns
template<typename RegT>
__device__ inline fp16* register_as_fp16(RegT* reg_ptr) {
    return register_cast<fp16>(reg_ptr);
}

template<typename RegT>
__device__ inline fp32* register_as_fp32(RegT* reg_ptr) {
    return register_cast<fp32>(reg_ptr);
}

template<typename RegT>
__device__ inline uint32_t* register_as_uint32(RegT* reg_ptr) {
    return register_cast<uint32_t>(reg_ptr);
}

// ========================= Vectorized Compute Patterns =========================

// Vectorized softmax computation pattern
template<typename T, size_t N>
struct softmax_compute_helper {
    // Apply scale and subtract max: result[i] = exp((input[i] * scale) - max_val)
    __device__ static void scale_sub_max_exp(const T input[N], T result[N], T scale, T max_val) {
        #pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = compute_exp<T>(compute_fma<T>(input[i], scale, -max_val));
        }
    }
    
    // Apply final normalization: result[i] = input[i] * scale_factor
    __device__ static void final_normalize(T input[N], T scale_factor) {
        #pragma unroll
        for (size_t i = 0; i < N; ++i) {
            input[i] = mul(input[i], scale_factor);
        }
    }
};

// Specialized softmax computation for mixed precision (FP16 input, FP32 computation)
template<size_t N>
struct softmax_compute_helper<fp16, N> {
    __device__ static void scale_sub_max_exp(const fp16 input[N], fp32 result[N], fp32 scale, fp32 max_val) {
        #pragma unroll
        for (size_t i = 0; i < N; ++i) {
            fp32 input_f32 = compute_cast<fp16, fp32>(input[i]);
            result[i] = compute_exp<fp32>(compute_fma<fp32>(input_f32, scale, -max_val));
        }
    }
    
    __device__ static void final_normalize_to_fp16(const fp32 input[N], fp16 result[N], fp32 scale_factor) {
        #pragma unroll
        for (size_t i = 0; i < N; ++i) {
            fp32 normalized = mul(input[i], scale_factor);
            result[i] = compute_cast<fp32, fp16>(normalized);
        }
    }
};

// Unified softmax computation API
template<typename T, size_t N>
__device__ inline void compute_softmax_scaled(const T input[N], T result[N], T scale, T max_val) {
    softmax_compute_helper<T, N>::scale_sub_max_exp(input, result, scale, max_val);
}

template<typename T, size_t N>
__device__ inline void compute_normalize(T input[N], T scale_factor) {
    softmax_compute_helper<T, N>::final_normalize(input, scale_factor);
}

// Mixed precision softmax computation
template<size_t N>
__device__ inline void compute_softmax_scaled_mixed(const fp16 input[N], fp32 result[N], fp32 scale, fp32 max_val) {
    softmax_compute_helper<fp16, N>::scale_sub_max_exp(input, result, scale, max_val);
}

template<size_t N>
__device__ inline void compute_normalize_to_fp16(const fp32 input[N], fp16 result[N], fp32 scale_factor) {
    softmax_compute_helper<fp16, N>::final_normalize_to_fp16(input, result, scale_factor);
}

// ========================= Accumulator Update Patterns =========================

// Accumulator rescaling pattern: acc = scale_factor * old_acc + new_value
template<typename T>
__device__ inline T compute_rescale_accumulate(T old_acc, T new_value, T scale_factor) {
    return compute_fma<T>(scale_factor, old_acc, new_value);
}

// Vectorized accumulator rescaling
template<typename T, size_t N>
__device__ inline void compute_rescale_accumulate_vec(T old_acc[N], const T new_value[N], T scale_factor) {
    #pragma unroll
    for (size_t i = 0; i < N; ++i) {
        old_acc[i] = compute_rescale_accumulate(old_acc[i], new_value[i], scale_factor);
    }
}

// Mixed precision accumulator rescaling (FP16 to FP32 accumulation)
template<size_t N>
__device__ inline void compute_rescale_accumulate_mixed(fp32 old_acc[N], const fp16 new_value[N], fp32 scale_factor) {
    #pragma unroll
    for (size_t i = 0; i < N; ++i) {
        fp32 new_value_f32 = compute_cast<fp16, fp32>(new_value[i]);
        old_acc[i] = compute_rescale_accumulate(old_acc[i], new_value_f32, scale_factor);
    }
}

// ========================= Register Element Access Helpers =========================

// Safe element access for register arrays with type conversion
template<typename ElementT, typename RegT, size_t RegIdx>
__device__ inline ElementT register_element_get(const RegT reg_array[], size_t element_idx) {
    const ElementT* element_ptr = register_cast<const ElementT>(&reg_array[RegIdx]);
    return element_ptr[element_idx];
}

template<typename ElementT, typename RegT, size_t RegIdx>
__device__ inline void register_element_set(RegT reg_array[], size_t element_idx, ElementT value) {
    ElementT* element_ptr = register_cast<ElementT>(&reg_array[RegIdx]);
    element_ptr[element_idx] = value;
}

// Convenience functions for common element access patterns
template<typename RegT>
__device__ inline fp16 register_get_fp16(const RegT reg_array[], size_t reg_idx, size_t element_idx) {
    return register_element_get<fp16, RegT, 0>(reg_array + reg_idx, element_idx);
}

template<typename RegT>
__device__ inline void register_set_fp16(RegT reg_array[], size_t reg_idx, size_t element_idx, fp16 value) {
    register_element_set<fp16, RegT, 0>(reg_array + reg_idx, element_idx, value);
}

template<typename RegT>
__device__ inline fp32 register_get_fp32(const RegT reg_array[], size_t reg_idx, size_t element_idx) {
    return register_element_get<fp32, RegT, 0>(reg_array + reg_idx, element_idx);
}

template<typename RegT>
__device__ inline void register_set_fp32(RegT reg_array[], size_t reg_idx, size_t element_idx, fp32 value) {
    register_element_set<fp32, RegT, 0>(reg_array + reg_idx, element_idx, value);
}

// ========================= Tile-Level Compute Primitives =========================

// Compile-time switch for verification mode  
#ifndef TILE_SERIAL_VERIFY
#define TILE_SERIAL_VERIFY 0
#endif

// ========================= Micro-Tile Level: Basic Building Blocks =========================

// Generic micro-tile load: Shared Memory -> Register
template<typename T, TileInt TileM, TileInt TileN, TileInt TileK, Layout LayoutType>
struct microtile_load_dispatcher {
    template<typename RegContainer>
    __device__ static void execute(
        SharedPtr<T> smem_ptr,
        RegContainer& reg_container, 
        IndexInt smem_stride,
        IndexInt lane_id,
        IndexInt tile_offset_m,
        IndexInt tile_offset_n,
        IndexInt tile_offset_k
    );
};

// Specialization for MMA-compatible loads
template<>
struct microtile_load_dispatcher<fp16, 16, 8, 16, Layout::ROW_MAJOR> {
    template<typename RegContainer>
    __device__ static void execute(
        SharedPtr<fp16> smem_ptr,
        RegContainer& reg_container,
        IndexInt smem_stride,
        IndexInt lane_id,
        IndexInt tile_offset_m, 
        IndexInt tile_offset_n,
        IndexInt tile_offset_k
    ) {
        IndexInt lane_m = tile_offset_m + lane_id % 16;
        IndexInt lane_k = tile_offset_k + (lane_id / 16) * 8;
        SmemAddr smem_addr = generic_to_shared_addr(&smem_ptr[lane_m * smem_stride + lane_k]);
        warp_copy_sync<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(
            smem_addr, reg_container[0], reg_container[1], reg_container[2], reg_container[3]);
    }
};

// Generic micro-tile compute: Register x Register -> Register
template<typename T, TileInt TileM, TileInt TileN, TileInt TileK, MmaLayout LayoutType>
struct microtile_mma_dispatcher {
    template<typename RegA, typename RegB, typename RegC>
    __device__ static void execute(RegA& reg_a, RegB& reg_b, RegC& reg_c);
};

// Specialization for standard MMA
template<>
struct microtile_mma_dispatcher<fp16, 16, 8, 16, MmaLayout::TN> {
    template<typename RegA, typename RegB, typename RegC>
    __device__ static void execute(RegA& reg_a, RegB& reg_b, RegC& reg_c) {
        mma<MmaShape::M16N8K16, MmaLayout::TN, fp16, fp16>(
            reg_c[0], reg_c[1], reg_a[0], reg_a[1], reg_a[2], reg_a[3], 
            reg_b[0], reg_b[1], reg_c[0], reg_c[1]);
    }
};

// ========================= Implementation Functions (defined first) =========================

// Parallel implementation for generic tile GEMM
template<typename T, 
         TileInt WarpTileM, TileInt WarpTileN, TileInt WarpTileK,
         TileInt MmaAtomM, TileInt MmaAtomN, TileInt MmaAtomK,
         TileInt MemStrideM, TileInt MemStrideN, TileInt MemStrideK,
         MmaLayout MmaLayoutType>
__device__ inline void tile_gemm_parallel(
    SharedPtr<T> A_smem, SharedPtr<T> B_smem,
    auto& C_regs, auto& A_regs, auto& B_regs,
    IndexInt k_tile_idx, IndexInt warp_m, IndexInt warp_n, IndexInt lane_id
) {
    // Load A tiles from Shared to Registers
    serial_range_for(i, 0, WarpTileM, 1) {
        microtile_load_dispatcher<T, MmaAtomM, MmaAtomN, MmaAtomK, Layout::ROW_MAJOR>::execute(
            A_smem, A_regs[i], MemStrideM, lane_id,
            warp_m * MmaAtomM + i * MmaAtomM, 0, k_tile_idx * MmaAtomK);
    }
    
    // Load B tiles from Shared to Registers
    serial_range_for(j, 0, WarpTileN, 1) {
        microtile_load_dispatcher<T, MmaAtomN, MmaAtomK, MmaAtomN, Layout::ROW_MAJOR>::execute(
            B_smem, B_regs[j], MemStrideN, lane_id,
            warp_n * MmaAtomN + j * MmaAtomN, 0, k_tile_idx * MmaAtomK);
    }
    
    // Perform MMA operations
    serial_range_for(i, 0, WarpTileM, 1) {
        serial_range_for(j, 0, WarpTileN, 1) {
            microtile_mma_dispatcher<T, MmaAtomM, MmaAtomN, MmaAtomK, MmaLayoutType>::execute(
                A_regs[i], B_regs[j], C_regs[i][j]);
        }
    }
}

// Serial implementation for verification
template<typename T, 
         TileInt WarpTileM, TileInt WarpTileN, TileInt WarpTileK,
         TileInt MmaAtomM, TileInt MmaAtomN, TileInt MmaAtomK,
         TileInt MemStrideK>
__device__ inline void tile_gemm_serial(
    SharedPtr<T> A_smem, SharedPtr<T> B_smem,
    auto& C_regs, IndexInt k_tile_idx
) {
    // Only thread 0 performs computation
    if (threadIdx.x != 0) return;
    
    constexpr TileInt TileM = WarpTileM * MmaAtomM;
    constexpr TileInt TileN = WarpTileN * MmaAtomN;
    constexpr TileInt TileK = MmaAtomK;
    
    // Serial matrix multiplication for verification
    mem_alloc_register(fp32, serial_C, [TileM][TileN]);
    mem_fill(serial_C, 0.0f);
    
    // Simple triple loops
    serial_range_for(m, 0, TileM, 1) {
        serial_range_for(n, 0, TileN, 1) {
            fp32 acc = 0.0f;
            serial_range_for(k, 0, TileK, 1) {
                IndexInt a_idx = m * MemStrideK + k_tile_idx * MmaAtomK + k;
                IndexInt b_idx = n * MemStrideK + k_tile_idx * MmaAtomK + k;
                fp32 a_val = compute_cast<T, fp32>(A_smem[a_idx]);
                fp32 b_val = compute_cast<T, fp32>(B_smem[b_idx]);
                acc = compute_fma<fp32>(a_val, b_val, acc);
            }
            serial_C[m][n] = acc;
        }
    }
    
    // Map results back to register format (simplified)
    serial_range_for(i, 0, WarpTileM, 1) {
        serial_range_for(j, 0, WarpTileN, 1) {
            fp16* reg_ptr = register_as_fp16(&C_regs[i][j][0]);
            IndexInt m_start = i * MmaAtomM;
            IndexInt n_start = j * MmaAtomN;
            
            // Simplified mapping for verification
            reg_ptr[0] = compute_cast<fp32, fp16>(serial_C[m_start + 0][n_start + 0]);
            reg_ptr[1] = compute_cast<fp32, fp16>(serial_C[m_start + 0][n_start + 1]);
            reg_ptr[2] = compute_cast<fp32, fp16>(serial_C[m_start + 8][n_start + 0]);
            reg_ptr[3] = compute_cast<fp32, fp16>(serial_C[m_start + 8][n_start + 1]);
        }
    }
    
    block_sync();
}

// ========================= Tile Level: Composed Operations =========================

// Generic tile-level GEMM operation
template<typename T, 
         TileInt WarpTileM, TileInt WarpTileN, TileInt WarpTileK,
         TileInt MmaAtomM, TileInt MmaAtomN, TileInt MmaAtomK,
         TileInt MemStrideM, TileInt MemStrideN, TileInt MemStrideK,
         MmaLayout MmaLayoutType = MmaLayout::TN>
__device__ inline void tile_gemm(
    SharedPtr<T> A_smem,         // Input A matrix in shared memory
    SharedPtr<T> B_smem,         // Input B matrix in shared memory  
    auto& C_regs,                // Output C accumulator registers
    auto& A_regs,                // Temporary A registers
    auto& B_regs,                // Temporary B registers
    IndexInt k_tile_idx,         // Current K dimension tile index
    IndexInt warp_m, IndexInt warp_n,  // Warp position in tile
    IndexInt lane_id             // Lane ID within warp
) {
#if TILE_SERIAL_VERIFY
    // Serial verification: only thread 0 computes using naive loops
    tile_gemm_serial<T, WarpTileM, WarpTileN, WarpTileK, MmaAtomM, MmaAtomN, MmaAtomK, MemStrideK>(
        A_smem, B_smem, C_regs, k_tile_idx);
#else
    // Parallel implementation using hardware accelerated primitives
    tile_gemm_parallel<T, WarpTileM, WarpTileN, WarpTileK, MmaAtomM, MmaAtomN, MmaAtomK, 
                       MemStrideM, MemStrideN, MemStrideK, MmaLayoutType>(
        A_smem, B_smem, C_regs, A_regs, B_regs, k_tile_idx, warp_m, warp_n, lane_id);
#endif
}

// ========================= Pattern-Specific Specializations =========================

// Flash Attention QK^T pattern - specialized implementation
template<TileInt WarpTileSeqQ, TileInt WarpTileSeqK, TileInt HeadDim>
__device__ inline void tile_flash_qkt(
    SharedPtr<fp16> Q_smem, SharedPtr<fp16> K_smem,
    auto& S_regs, auto& Q_regs, auto& K_regs,
    IndexInt k_tile_idx, IndexInt warp_qp, IndexInt warp_kv, IndexInt lane_id
) {
    SmemAddr smem_Q_base_addr = generic_to_shared_addr(Q_smem);
    SmemAddr smem_K_base_addr = generic_to_shared_addr(K_smem);
    
    // Load Q tile from Shared to Registers (Flash Attention specific pattern)
    serial_range_for(i, 0, WarpTileSeqQ, 1) {
        IndexInt warp_smem_Q_Br = warp_qp * (16 * WarpTileSeqQ) + i * 16;
        IndexInt lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
        IndexInt lane_smem_Q_d = k_tile_idx * 16 + (lane_id / 16) * 8;
        SmemAddr lane_smem_Q_addr = smem_Q_base_addr + (lane_smem_Q_Br * HeadDim + lane_smem_Q_d) * sizeof(fp16);
        warp_copy_sync<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(lane_smem_Q_addr, Q_regs[i][0], Q_regs[i][1], Q_regs[i][2], Q_regs[i][3]);
    }

    // Load K tile from Shared to Registers (Flash Attention specific pattern)
    serial_range_for(j, 0, WarpTileSeqK, 1) {
        IndexInt warp_smem_K_Bc = j * 8;
        IndexInt lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8;
        IndexInt lane_smem_K_d = k_tile_idx * 16 + ((lane_id / 8) % 2) * 8;
        SmemAddr lane_smem_K_addr = smem_K_base_addr + (lane_smem_K_Bc * HeadDim + lane_smem_K_d) * sizeof(fp16);
        warp_copy_sync<fp16, 64, CopyTask::S2R, Layout::ROW_MAJOR>(lane_smem_K_addr, K_regs[j][0], K_regs[j][1]);
    }

    // Perform MMA for QK^T
    serial_range_for(j, 0, WarpTileSeqK, 1) {
        mma<MmaShape::M16N8K16, MmaLayout::TN, fp16, fp16>(S_regs[0][j][0], S_regs[0][j][1], Q_regs[0][0], Q_regs[0][1], Q_regs[0][2], Q_regs[0][3], K_regs[j][0], K_regs[j][1], S_regs[0][j][0], S_regs[0][j][1]);
    }
}

// Flash Attention PV pattern  
template<TileInt WarpTileSeqP, TileInt WarpTileHeadV, TileInt SeqLen>
__device__ inline void tile_flash_pv(
    SharedPtr<fp16> P_regs, SharedPtr<fp16> V_smem, 
    auto& O_regs, auto& P_temp_regs, auto& V_regs,
    IndexInt v_tile_idx, IndexInt warp_p, IndexInt warp_v, IndexInt lane_id
) {
    // Use generic tile_gemm with P@V specific parameters
    tile_gemm<fp16, WarpTileSeqP, WarpTileHeadV, 1, 16, 8, 16, SeqLen, 64, 16, MmaLayout::TN>(
        P_regs, V_smem, O_regs, P_temp_regs, V_regs, v_tile_idx, warp_p, warp_v, lane_id);
}

// Standard GEMM pattern
template<TileInt WarpM, TileInt WarpN, TileInt WarpK>  
__device__ inline void tile_standard_gemm(
    SharedPtr<fp16> A_smem, SharedPtr<fp16> B_smem,
    auto& C_regs, auto& A_regs, auto& B_regs,
    IndexInt k_tile_idx, IndexInt warp_m, IndexInt warp_n, IndexInt lane_id
) {
    tile_gemm<fp16, WarpM, WarpN, WarpK, 16, 8, 16, WarpM*16, WarpN*8, 16, MmaLayout::TN>(
        A_smem, B_smem, C_regs, A_regs, B_regs, k_tile_idx, warp_m, warp_n, lane_id);
}



// ========================= Higher-Level Compute Patterns =========================

// Tile-level softmax pattern
template<TileInt WarpTileSeq, TileInt AtomN>
__device__ inline void tile_softmax(
    auto& S_regs,           // Input scores, output probabilities  
    fp32 scale,             // Attention scale factor
    auto& max_regs,         // Max value registers
    auto& sum_regs,         // Sum value registers
    IndexInt lane_id
) {
    // Find local max across registers
    serial_range_for(j, 0, WarpTileSeq, 1) {
        fp16* s_ptr = register_as_fp16(&S_regs[0][j][0]);
        fp32 local_max_0 = compute_cast<fp16, fp32>(compute_max<fp16>(s_ptr[0], s_ptr[1])) * scale;
        fp32 local_max_1 = compute_cast<fp16, fp32>(compute_max<fp16>(s_ptr[2], s_ptr[3])) * scale;
        max_regs[0][0] = compute_max<fp32>(max_regs[0][0], local_max_0);
        max_regs[0][1] = compute_max<fp32>(max_regs[0][1], local_max_1);
    }
    
    // Warp reduction for max (4-thread sub-warp for Flash Attention)
    max_regs[0][0] = warp_shuffle_max_width<fp32>(max_regs[0][0], 4);
    max_regs[0][1] = warp_shuffle_max_width<fp32>(max_regs[0][1], 4);
    
    // Compute exponentials and sum
    serial_range_for(j, 0, WarpTileSeq, 1) {
        fp16* s_ptr = register_as_fp16(&S_regs[0][j][0]);
        fp32_4 exp_vals;
        exp_vals.x = compute_exp<fp32>(compute_fma<fp32>(compute_cast<fp16, fp32>(s_ptr[0]), scale, -max_regs[0][0]));
        exp_vals.y = compute_exp<fp32>(compute_fma<fp32>(compute_cast<fp16, fp32>(s_ptr[1]), scale, -max_regs[0][0]));
        exp_vals.z = compute_exp<fp32>(compute_fma<fp32>(compute_cast<fp16, fp32>(s_ptr[2]), scale, -max_regs[0][1]));
        exp_vals.w = compute_exp<fp32>(compute_fma<fp32>(compute_cast<fp16, fp32>(s_ptr[3]), scale, -max_regs[0][1]));
        
        sum_regs[0][0] += (exp_vals.x + exp_vals.y);
        sum_regs[0][1] += (exp_vals.z + exp_vals.w);
        
        s_ptr[0] = compute_cast<fp32, fp16>(exp_vals.x);
        s_ptr[1] = compute_cast<fp32, fp16>(exp_vals.y);
        s_ptr[2] = compute_cast<fp32, fp16>(exp_vals.z);
        s_ptr[3] = compute_cast<fp32, fp16>(exp_vals.w);
    }
    
    // Warp reduction for sum (4-thread sub-warp)
    sum_regs[0][0] = warp_shuffle_sum_width<fp32>(sum_regs[0][0], 4);
    sum_regs[0][1] = warp_shuffle_sum_width<fp32>(sum_regs[0][1], 4);
}

// Tile-level accumulator rescaling pattern
template<TileInt WarpTileM, TileInt WarpTileN>
__device__ inline void tile_rescale_accumulator(
    auto& O_regs,           // Accumulator registers
    auto& D_regs,           // Output registers
    fp32 rescale_factor_0,  // Rescale factor for first half
    fp32 rescale_factor_1   // Rescale factor for second half
) {
    serial_range_for(j, 0, WarpTileN, 1) {
        fp16* o_ptr = register_as_fp16(&O_regs[0][j][0]);
        fp32* d_ptr = register_as_fp32(&D_regs[0][j][0]);
        
        d_ptr[0] = compute_rescale_accumulate<fp32>(d_ptr[0], compute_cast<fp16, fp32>(o_ptr[0]), rescale_factor_0);
        d_ptr[1] = compute_rescale_accumulate<fp32>(d_ptr[1], compute_cast<fp16, fp32>(o_ptr[1]), rescale_factor_0);
        d_ptr[2] = compute_rescale_accumulate<fp32>(d_ptr[2], compute_cast<fp16, fp32>(o_ptr[2]), rescale_factor_1);
        d_ptr[3] = compute_rescale_accumulate<fp32>(d_ptr[3], compute_cast<fp16, fp32>(o_ptr[3]), rescale_factor_1);
    }
}

// Tile-level result redistribution pattern
template<TileInt WarpTileN>
__device__ inline void tile_redistribute_results(
    auto& src_regs,         // Source registers
    auto& dst_regs,         // Destination registers
    IndexInt lane_id
) {
    serial_range_for(j, 0, WarpTileN, 1) {
        warp_shuffle_spread_dual_x4(src_regs[j][0], src_regs[j][1], 
                                    dst_regs[0], dst_regs[1], lane_id, 4);
    }
}

// ========================= PyTorch Integration Primitives =========================

// Helper to stringify macro arguments
#define STRINGIFY(X) #X
#define STRINGFY(str) #str

// Tensor validation macros
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type); \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
    throw std::runtime_error("Tensor size mismatch!"); \
  }

// PyTorch binding macro
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGIFY(func), &func, STRINGIFY(func));
