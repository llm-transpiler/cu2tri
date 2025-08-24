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
    return static_cast<SharedPtr<T>>(raw_ptr);
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

// Template-based copy dispatcher for thread-level operations
template<typename T, size_t BitWidth, CopyTask Task, bool IsAsync = false>
struct copy_dispatcher {
    template<typename SrcPtr, typename DstPtr>
    __device__ static void execute(SrcPtr src_ptr, DstPtr dst_ptr) {
        constexpr size_t bytes = BitWidth / 8;
        
        // Async copy operations (only for G2S and S2G)
        if constexpr (IsAsync) {
            if constexpr (Task == CopyTask::G2S) {
                asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" 
                    :: "r"(dst_ptr), "l"(src_ptr), "n"(bytes));
            } else if constexpr (Task == CopyTask::S2G) {
                // S2G async copy uses same instruction (implementation may vary)
                asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" 
                    :: "r"(dst_ptr), "l"(src_ptr), "n"(bytes));
            }
        } 
        // Synchronous copy operations
        else {
            if constexpr (BitWidth == 128) {
                (reinterpret_cast<float4 *>(dst_ptr)[0]) = (reinterpret_cast<float4 *>(src_ptr)[0]);
            } else if constexpr (BitWidth == 64) {
                (reinterpret_cast<float2 *>(dst_ptr)[0]) = (reinterpret_cast<float2 *>(src_ptr)[0]);
            } else if constexpr (BitWidth == 32) {
                (reinterpret_cast<half2 *>(dst_ptr)[0]) = (reinterpret_cast<half2 *>(src_ptr)[0]);
            }
        }
    }
};

// Thread-level synchronous copy template
template<typename T, size_t BitWidth, CopyTask Task, typename SrcPtr, typename DstPtr>
__device__ inline void thread_copy_template(SrcPtr src_ptr, DstPtr dst_ptr) {
    copy_dispatcher<T, BitWidth, Task, false>::execute(src_ptr, dst_ptr);
}

// Thread-level asynchronous copy template  
template<typename T, size_t BitWidth, CopyTask Task, typename SrcPtr, typename DstPtr>
__device__ inline void thread_copy_async_template(SrcPtr src_ptr, DstPtr dst_ptr) {
    copy_dispatcher<T, BitWidth, Task, true>::execute(src_ptr, dst_ptr);
}

// Clean template-based API - all parameters as template arguments
template<typename T, size_t BitWidth, CopyTask Task, typename SrcPtr, typename DstPtr>
__device__ inline void thread_copy(SrcPtr src_ptr, DstPtr dst_ptr) {
    copy_dispatcher<T, BitWidth, Task, false>::execute(src_ptr, dst_ptr);
}

template<typename T, size_t BitWidth, CopyTask Task, typename SrcPtr, typename DstPtr>
__device__ inline void thread_copy_async(SrcPtr src_ptr, DstPtr dst_ptr) {
    if constexpr (Task == CopyTask::G2S || Task == CopyTask::S2G) {
        copy_dispatcher<T, BitWidth, Task, true>::execute(src_ptr, dst_ptr);
    } else {
        // Async copy only supported for G2S and S2G, others fall back to sync
        copy_dispatcher<T, BitWidth, Task, false>::execute(src_ptr, dst_ptr);
    }
}

// Async copy control functions
__device__ inline void thread_copy_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ inline void thread_copy_async_wait_group() {
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

// Warp-level copy dispatcher
template<typename T, size_t BitWidth, CopyTask Task, Layout LayoutType>
struct warp_copy_dispatcher {
    template<typename SrcPtr, typename... Args>
    __device__ static void execute(SrcPtr src_ptr, Args&... args) {
        if constexpr (Task == CopyTask::S2R) {
            // Shared to Register - ldmatrix instructions
            SmemAddr s_src_ptr = __cvta_generic_to_shared(src_ptr);
            
            if constexpr (LayoutType == Layout::ROW_MAJOR) {
                if constexpr (BitWidth == 32 && sizeof...(args) == 1) {
                    LDMATRIX_X1(get_arg<0>(args...), s_src_ptr);
                } else if constexpr (BitWidth == 64 && sizeof...(args) == 2) {
                    LDMATRIX_X2(get_arg<0>(args...), get_arg<1>(args...), s_src_ptr);
                } else if constexpr (BitWidth == 128 && sizeof...(args) == 4) {
                    LDMATRIX_X4(get_arg<0>(args...), get_arg<1>(args...), get_arg<2>(args...), get_arg<3>(args...), s_src_ptr);
                }
            } else if constexpr (LayoutType == Layout::COL_MAJOR) {
                if constexpr (BitWidth == 32 && sizeof...(args) == 1) {
                    LDMATRIX_X1_T(get_arg<0>(args...), s_src_ptr);
                } else if constexpr (BitWidth == 64 && sizeof...(args) == 2) {
                    LDMATRIX_X2_T(get_arg<0>(args...), get_arg<1>(args...), s_src_ptr);
                } else if constexpr (BitWidth == 128 && sizeof...(args) == 4) {
                    LDMATRIX_X4_T(get_arg<0>(args...), get_arg<1>(args...), get_arg<2>(args...), get_arg<3>(args...), s_src_ptr);
                }
            }
        } else if constexpr (Task == CopyTask::R2S) {
            // Register to Shared - stmatrix instructions (SM90+)
            SmemAddr s_dst_ptr = __cvta_generic_to_shared(src_ptr);
            
            if constexpr (LayoutType == Layout::ROW_MAJOR) {
                if constexpr (BitWidth == 32 && sizeof...(args) == 1) {
                    STMATRIX_X1(s_dst_ptr, get_arg<0>(args...));
                } else if constexpr (BitWidth == 64 && sizeof...(args) == 2) {
                    STMATRIX_X2(s_dst_ptr, get_arg<0>(args...), get_arg<1>(args...));
                } else if constexpr (BitWidth == 128 && sizeof...(args) == 4) {
                    STMATRIX_X4(s_dst_ptr, get_arg<0>(args...), get_arg<1>(args...), get_arg<2>(args...), get_arg<3>(args...));
                }
            } else if constexpr (LayoutType == Layout::COL_MAJOR) {
                if constexpr (BitWidth == 32 && sizeof...(args) == 1) {
                    STMATRIX_X1_T(s_dst_ptr, get_arg<0>(args...));
                } else if constexpr (BitWidth == 64 && sizeof...(args) == 2) {
                    STMATRIX_X2_T(s_dst_ptr, get_arg<0>(args...), get_arg<1>(args...));
                } else if constexpr (BitWidth == 128 && sizeof...(args) == 4) {
                    STMATRIX_X4_T(s_dst_ptr, get_arg<0>(args...), get_arg<1>(args...), get_arg<2>(args...), get_arg<3>(args...));
                }
            }
        }
        // Note: Other CopyTask values (G2R, etc.) would need different instruction implementations
    }
};

// Clean template-based warp copy API - all parameters as template arguments
template<typename T, size_t BitWidth, CopyTask Task, Layout LayoutType, typename SrcPtr, typename... Args>
__device__ inline void warp_copy(SrcPtr src_ptr, Args&... args) {
    warp_copy_dispatcher<T, BitWidth, Task, LayoutType>::execute(src_ptr, args...);
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

// Warp shuffle operations
template<typename T>
__device__ inline T warp_shuffle(T value, int src_lane, unsigned mask = 0xffffffff) {
    return __shfl_sync(mask, value, src_lane);
}

// Warp shuffle with offset
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

// ========================= Advanced Warp Shuffle Patterns =========================

// MMA result redistribution - spread single value to consecutive lanes
template<typename T>
__device__ inline void warp_shuffle_spread_x4(T value, T output[4], IndexInt lane_id, unsigned mask = 0xffffffff) {
    output[0] = value;                                        // Current lane
    output[1] = __shfl_sync(mask, value, lane_id + 1);       // Next lane
    output[2] = __shfl_sync(mask, value, lane_id + 2);       // Lane + 2  
    output[3] = __shfl_sync(mask, value, lane_id + 3);       // Lane + 3
}

// MMA result redistribution for dual accumulator pattern (common in GEMM)
template<typename T>
__device__ inline void warp_shuffle_spread_dual_x4(T src0, T src1, T dst0[4], T dst1[4], IndexInt lane_id, unsigned mask = 0xffffffff) {
    warp_shuffle_spread_x4(src0, dst0, lane_id, mask);
    warp_shuffle_spread_x4(src1, dst1, lane_id, mask);
}

// Broadcast value from specific lane to all lanes in warp
template<typename T>
__device__ inline T warp_shuffle_broadcast(T value, IndexInt src_lane, unsigned mask = 0xffffffff) {
    return __shfl_sync(mask, value, src_lane);
}

// ========================= Warp Reduction Primitives =========================

// Generic warp reduction using butterfly shuffle (XOR-based)
template<typename T, typename BinaryOp>
__device__ inline T warp_shuffle_reduce(T value, BinaryOp op, unsigned mask = 0xffffffff) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_xor_sync(mask, value, offset);
        value = op(value, other);
    }
    return value;
}

// Specialized reductions for different data types
template<typename T>
__device__ inline T warp_shuffle_sum(T value, unsigned mask = 0xffffffff);

// FP32 sum reduction
template<>
__device__ inline fp32 warp_shuffle_sum<fp32>(fp32 value, unsigned mask) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(mask, value, offset);
    }
    return value;
}

// FP16 sum reduction
template<>
__device__ inline fp16 warp_shuffle_sum<fp16>(fp16 value, unsigned mask) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value = __hadd(value, __shfl_xor_sync(mask, value, offset));
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

// Generic max reduction
template<typename T>
__device__ inline T warp_shuffle_max(T value, unsigned mask = 0xffffffff) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_xor_sync(mask, value, offset);
        value = value > other ? value : other;
    }
    return value;
}

// FP32 max reduction (using fmaxf)
template<>
__device__ inline fp32 warp_shuffle_max<fp32>(fp32 value, unsigned mask) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(mask, value, offset));
    }
    return value;
}

// Generic min reduction
template<typename T>
__device__ inline T warp_shuffle_min(T value, unsigned mask = 0xffffffff) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_xor_sync(mask, value, offset);
        value = value < other ? value : other;
    }
    return value;
}

// FP32 min reduction (using fminf)
template<>
__device__ inline fp32 warp_shuffle_min<fp32>(fp32 value, unsigned mask) {
    #pragma unroll
    for (IndexInt offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value = fminf(value, __shfl_xor_sync(mask, value, offset));
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

// Macro for backward compatibility
#define warp_shuffle(value, src_lane) warp_shuffle(value, src_lane)

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
