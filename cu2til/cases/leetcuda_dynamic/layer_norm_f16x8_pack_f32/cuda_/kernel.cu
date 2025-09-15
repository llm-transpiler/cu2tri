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
// -------------------------------------- Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc.
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  val = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0)
    shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val = warp_reduce_sum_f32<NUM_WARPS>(val);
  return val;
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_pack_f32_kernel(half *x, half *y, float g,
                                                 float b, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x;  // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 8;
  const float epsilon = 1e-5f;

  __shared__ float s_mean;     // shared within block
  __shared__ float s_variance; // shared within block
  // temporary register(memory), .local space in ptx, addressable
  half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

  float value = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    value += ((idx + i) < N * K ? __half2float(pack_x[i]) : 0.0f);
  }
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  if (tid == 0)
    s_mean = sum / (float)K;
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();

  float variance = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float v_hat = __half2float(pack_x[i]) - s_mean;
    variance += ((idx + i) < N * K ? v_hat * v_hat : 0.0f);
  }
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    pack_y[i] = __float2half(
        __fmaf_rn(((__half2float(pack_x[i]) - s_mean) * s_variance), g, b));
  }
  // reinterpret as float4 and store 128 bits in 1 memory issue.
  if ((idx + 7) < N * K) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
  // TODO: support non 8-multiple K here
}

extern "C" void cuda_kernel(half* x, half* y, float g, float b, int N, int K) {
    dim3 block(K / 8);
    dim3 grid(N);
    switch (K) {
    case 64:
        layer_norm_f16x8_pack_f32_kernel<8><<<grid, block>>>(x, y, g, b, N, K);
        break;
    case 128:
        layer_norm_f16x8_pack_f32_kernel<16><<<grid, block>>>(x, y, g, b, N, K);
        break;
    case 256:
        layer_norm_f16x8_pack_f32_kernel<32><<<grid, block>>>(x, y, g, b, N, K);
        break;
    case 512:
        layer_norm_f16x8_pack_f32_kernel<64><<<grid, block>>>(x, y, g, b, N, K);
        break;
    case 1024:
        layer_norm_f16x8_pack_f32_kernel<128><<<grid, block>>>(x, y, g, b, N, K);
        break;
    case 2048:
        layer_norm_f16x8_pack_f32_kernel<256><<<grid, block>>>(x, y, g, b, N, K);
        break;
    case 4096:
        layer_norm_f16x8_pack_f32_kernel<512><<<grid, block>>>(x, y, g, b, N, K);
        break;
    case 8192:
        layer_norm_f16x8_pack_f32_kernel<1024><<<grid, block>>>(x, y, g, b, N, K);
        break;
    default:
        printf("only support K: 64/128/256/512/1024/2048/4096/8192\n");
        break;
    }
}
