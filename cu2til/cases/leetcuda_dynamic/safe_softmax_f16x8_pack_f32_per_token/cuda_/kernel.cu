#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// -------------------------------------- FP32
// -------------------------------------- DS required for Online Softmax
struct __align__(8) MD {
  float m;
  float d;
};
// Warp Reduce for Online Softmax
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
  unsigned int mask = 0xffffffff;
#pragma unroll
  for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
    MD other;
    other.m = __shfl_xor_sync(mask, value.m, stride);
    other.d = __shfl_xor_sync(mask, value.d, stride);

    bool value_bigger = (value.m > other.m);
    MD bigger_m = value_bigger ? value : other;
    MD smaller_m = value_bigger ? other : value;

    value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    value.m = bigger_m.m;
  }
  return value;
}

// Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Warp Reduce Max
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// grid 1D block 1D, grid(N/256), block(256)
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  float value = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0)
    shared[warp] = value;
  __syncthreads();
  value = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  value = warp_reduce_sum_f32<NUM_WARPS>(value);
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

template <const int NUM_THREADS = 256>
__device__ float block_reduce_max_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  float value = warp_reduce_max_f32<WARP_SIZE>(val);
  if (lane == 0)
    shared[warp] = value;
  __syncthreads();
  value = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
  value = warp_reduce_max_f32<NUM_WARPS>(value);
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x8_pack_f32_per_token_kernel(half *x, half *y,
                                                             int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 8;
  // temporary register(memory), .local space in ptx, addressable
  half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

  float max_val = -FLT_MAX;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    max_val = fmaxf(__half2float(pack_x[i]), max_val);
  }
  max_val = block_reduce_max_f32<NUM_THREADS>(max_val); // block max

  float exp_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float exp_val = expf(__half2float(pack_x[i]) - max_val);
    exp_sum += (((idx + i) < N) ? exp_val : 0.0f);
  }
  exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_sum); // block sum

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    // e^x_i/sum(e^x_0,...,e^x_n-1)
    float exp_val = expf(__half2float(pack_x[i]) - max_val);
    pack_y[i] = __float2half_rn(exp_val / exp_sum);
  }
  // reinterpret as float4 and store 128 bits in 1 memory issue.
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
  // TODO: support non 8-multiple K here
}

#define LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(H)                 \
  safe_softmax_f16x8_pack_f32_per_token_kernel<(H) / 8>                        \
      <<<grid, block>>>(reinterpret_cast<half *>(x),                           \
                        reinterpret_cast<half *>(y), N);

#define DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H)            \
  const int NT = (H) / 8;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(32) break;             \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(64) break;             \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(128) break;            \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(256) break;            \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(512) break;            \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(1024) break;           \
  case 2048:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(2048) break;           \
  case 4096:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(4096) break;           \
  case 8192:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(8192) break;           \
  default:                                                                     \
    assert(false && "only support H: 64/128/.../1024*8");                      \
    break;                                                                     \
  }

extern "C" void cuda_kernel(half* x, half* y, const int S, const int H) {
    const int N = S * H;
    DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H)
}
