#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// HardSwish activation function for half: x * ReLU6(x + 3) / 6
__device__ __forceinline__ half hardswish_half(half x) {
  half three = __float2half(3.0f);
  half six = __float2half(6.0f);
  half temp = __hadd(x, three);
  // ReLU6: min(max(x, 0), 6)
  half relu6 = __hmax(__hmin(temp, six), __float2half(0.0f));
  return __hdiv(__hmul(x, relu6), six);
}

__global__ void hardswish_f16x8_pack_kernel(half *x, half *y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half pack_x[8], pack_y[8];
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
  for (int i = 0; i < 8; i++) {
    pack_y[i] = hardswish_half(pack_x[i]);
  }
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}

extern "C" void cuda_kernel(half* x, half* y, int N) {
    dim3 block(256 / 8);  // 32 threads
    dim3 grid((N + 256 - 1) / 256);
    hardswish_f16x8_pack_kernel<<<grid, block>>>(x, y, N);
}
