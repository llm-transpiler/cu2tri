#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// Warp Reduce Sum (half precision)
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// Block reduce sum device helper for RMS Norm (half precision)
template <const int NUM_THREADS = 64>
__device__ half block_reduce_sum_f16_f16(half val) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ half shared[NUM_WARPS];

  val = warp_reduce_sum_f16_f16<WARP_SIZE>(val);
  if (lane == 0)
    shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : __float2half(0.0f);
  val = warp_reduce_sum_f16_f16<NUM_WARPS>(val);
  return val;
}

// RMS Norm f16x8_pack -> f16 output
template <const int NUM_THREADS = 64>
__global__ void __launch_bounds__(NUM_THREADS)
    _cuda_kernel_impl(half *__restrict__ x, half *__restrict__ y, float g, int N, int K) {
  int tid = threadIdx.x; // 0..K/8-1
  int bid = blockIdx.x;  // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 8;
  const half epsilon = __float2half(1e-5f);
  const half g_ = __float2half(g);
  const half K_ = __int2half_rn(K);
  const half z_ = __float2half(0.0f);

  __shared__ half s_variance; // shared within block
  
  // temporary register(memory), .local space in ptx, addressable
  half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
  
  if (idx + 7 < N * K) {
    // reinterpret as float4 and load 128 bits in 1 memory issue.
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits
  } else {
    // Handle boundary case
    for (int i = 0; i < 8; ++i) {
      pack_x[i] = (idx + i < N * K) ? x[idx + i] : z_;
    }
  }

  half variance = z_;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    half val = ((idx + i) < N * K ? pack_x[i] : z_);
    variance = __hadd(variance, __hmul(val, val));
  }
  
  variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = __hmul(hrsqrt(__hadd(__hdiv(variance, K_), epsilon)), g_);
  
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();

  // Apply RMS normalization
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if ((idx + i) < N * K) {
      pack_y[i] = __hmul(pack_x[i], s_variance);
    } else {
      pack_y[i] = z_;
    }
  }
  
  if (idx + 7 < N * K) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]); // store 128 bits
  } else {
    // Handle boundary case
    for (int i = 0; i < 8; ++i) {
      if (idx + i < N * K) {
        y[idx + i] = pack_y[i];
      }
    }
  }
}

extern "C" void cuda_kernel(half *x, half *y, float g, int N, int K) {
  // Fixed size for this test case: N=8, K=512 (8 sequences, 512 features)
  // Each thread processes 8 elements, so we need K/8 threads per block
  
  dim3 block(K / 8);  // 512/8 = 64 threads per block
  dim3 grid(N);       // 8 blocks
  
  switch (K) {
    case 64:
      _cuda_kernel_impl<8><<<grid, block>>>(x, y, g, N, K);
      break;
    case 128:
      _cuda_kernel_impl<16><<<grid, block>>>(x, y, g, N, K);
      break;
    case 256:
      _cuda_kernel_impl<32><<<grid, block>>>(x, y, g, N, K);
      break;
    case 512:
      _cuda_kernel_impl<64><<<grid, block>>>(x, y, g, N, K);
      break;
    case 1024:
      _cuda_kernel_impl<128><<<grid, block>>>(x, y, g, N, K);
      break;
    case 2048:
      _cuda_kernel_impl<256><<<grid, block>>>(x, y, g, N, K);
      break;
    case 4096:
      _cuda_kernel_impl<512><<<grid, block>>>(x, y, g, N, K);
      break;
    case 8192:
      _cuda_kernel_impl<1024><<<grid, block>>>(x, y, g, N, K);
      break;
    default:
      printf("Only support K: 64/128/256/512/1024/2048/4096/8192\n");
      return;
  }
}
