#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
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

template <const int NUM_THREADS = 256 / 4>
__global__ void
online_safe_softmax_f32x4_pack_per_token_kernel(float *x, float *y, int N) {
  // reference: https://arxiv.org/pdf/1805.02867 (Online normalizer calculation
  // for softmax)
  int local_tid = threadIdx.x;
  int global_tid = (blockIdx.x * NUM_THREADS + local_tid) * 4;

  const int WAPR_NUM = NUM_THREADS / WARP_SIZE;
  int warp_id = local_tid / WARP_SIZE;
  int lane_id = local_tid % WARP_SIZE;
  // compare local max value
  float4 val = FLOAT4((x)[global_tid]);
  float local_m = fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w));
  float local_d = __expf(val.x - local_m) + __expf(val.y - local_m) +
                  __expf(val.z - local_m) + __expf(val.w - local_m);

  MD local_md = {local_m, local_d};
  MD res = warp_reduce_md_op<WARP_SIZE>(local_md);
  __shared__ MD shared[WAPR_NUM];

  if (lane_id == 0)
    shared[warp_id] = res;
  __syncthreads();
  // do block reduce
  if (local_tid < WARP_SIZE) {
    MD block_res = shared[local_tid];
    block_res = warp_reduce_md_op<WAPR_NUM>(block_res);
    if (local_tid == 0)
      shared[0] = block_res;
  }
  __syncthreads();
  // write back
  MD final_res = shared[0];
  float d_total_inverse = __fdividef(1.0f, final_res.d);
  if (global_tid < N) {
    float4 reg_y;
    reg_y.x = __expf(val.x - final_res.m) * d_total_inverse;
    reg_y.y = __expf(val.y - final_res.m) * d_total_inverse;
    reg_y.z = __expf(val.z - final_res.m) * d_total_inverse;
    reg_y.w = __expf(val.w - final_res.m) * d_total_inverse;
    FLOAT4((y)[global_tid]) = reg_y;
  }
}

// online softmax per token
#define LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(H)                   \
  online_safe_softmax_f32x4_pack_per_token_kernel<(H / 4)>                     \
      <<<grid, block>>>(reinterpret_cast<float *>(x),               \
                        reinterpret_cast<float *>(y), N);

#define DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H)              \
  dim3 block((H / 4));                                                         \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 128:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(128)                     \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(256)                     \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(512)                     \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(1024)                    \
    break;                                                                     \
  case 2048:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(2048)                    \
    break;                                                                     \
  case 4096:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(4096)                    \
    break;                                                                     \
  default:                                                                     \
    assert(false && "only support H: 128/256/.../4096;");                      \
    break;                                                                     \
  }

extern "C" void cuda_kernel(float* x, float* y, const int S, const int H) {
  const int N = S * H;
  DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H)
}
