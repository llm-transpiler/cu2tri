#include <algorithm>
#include <cuda_fp16.h>
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

// 定义全局 alpha 值
#define ALPHA 1.0f

// ELU 计算函数
// -------------------------------------- FP32
// --------------------------------------
__device__ __forceinline__ float elu(float x) {
  return x > 0.f ? x : ALPHA * (expf(x) - 1.f);
}

// CUDA 核函数
// -------------------------------------- FP32
// --------------------------------------
__global__ void elu_f32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = elu(reg_x.x);
    reg_y.y = elu(reg_x.y);
    reg_y.z = elu(reg_x.z);
    reg_y.w = elu(reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

extern "C" void cuda_kernel(float* x, float* y, int N) {
    dim3 block(256 / 4);  // 64 threads
    dim3 grid((N + 256 - 1) / 256);
    elu_f32x4_kernel<<<grid, block>>>(x, y, N);
}
