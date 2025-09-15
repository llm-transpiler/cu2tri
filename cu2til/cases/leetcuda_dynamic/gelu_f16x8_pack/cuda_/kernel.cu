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
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define SQRT_2_PI M_SQRT2 *M_2_SQRTPI * 0.5f

#define GELU_OPS gelu_tanh_approximate

__inline__ __device__ float gelu_tanh_approximate(float x) {
  return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715f * x * x * x)));
}

// GELU for half precision
__device__ __forceinline__ half gelu_half(half x) {
  float x_f32 = __half2float(x);
  x_f32 = fminf(fmaxf(x_f32, MIN_EXP_F32), MAX_EXP_F32);
  float result = GELU_OPS(x_f32);
  return __float2half(result);
}

__global__ void gelu_f16x8_pack_kernel(half *x, half *y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half pack_x[8], pack_y[8];
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
  for (int i = 0; i < 8; i++) {
    pack_y[i] = gelu_half(pack_x[i]);
  }
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}

extern "C" void cuda_kernel(half* x, half* y, int N) {
    dim3 block(256 / 8);  // 32 threads
    dim3 grid((N + 256 - 1) / 256);
    gelu_f16x8_pack_kernel<<<grid, block>>>(x, y, N);
}
