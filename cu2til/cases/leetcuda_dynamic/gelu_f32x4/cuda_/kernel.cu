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
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define SQRT_2_PI M_SQRT2 *M_2_SQRTPI * 0.5f

#define GELU_OPS gelu_tanh_approximate

__inline__ __device__ float gelu_tanh_approximate(float x) {
  return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715f * x * x * x)));
}

// GELU tanh approximate; Vec4
// grid(N/256), block(256/4)
__global__ void gelu_f32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_y;

  reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);

  reg_y.x = GELU_OPS(reg_x.x);
  reg_y.y = GELU_OPS(reg_x.y);
  reg_y.z = GELU_OPS(reg_x.z);
  reg_y.w = GELU_OPS(reg_x.w);

  if ((idx + 0) < N) {
    FLOAT4(y[idx]) = reg_y;
  }
}

extern "C" void cuda_kernel(float* x, float* y, int N) {
    dim3 block(256 / 4);  // 64 threads
    dim3 grid((N + 256 - 1) / 256);
    gelu_f32x4_kernel<<<grid, block>>>(x, y, N);
}
