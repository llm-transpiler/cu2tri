#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LAMBDA 0.5f  // Default threshold for hardshrink

// HardShrink activation function for half
__device__ __forceinline__ half hardshrink_half(half x) {
  half abs_x = __habs(x);
  half lambda_half = __float2half(LAMBDA);
  return __hgt(abs_x, lambda_half) ? x : __float2half(0.0f);
}

__global__ void hardshrink_f16x8_pack_kernel(half *x, half *y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half pack_x[8], pack_y[8];
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
  for (int i = 0; i < 8; i++) {
    pack_y[i] = hardshrink_half(pack_x[i]);
  }
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}

extern "C" void cuda_kernel(half* x, half* y, int N) {
    dim3 block(256 / 8);  // 32 threads
    dim3 grid((N + 256 - 1) / 256);
    hardshrink_f16x8_pack_kernel<<<grid, block>>>(x, y, N);
}
