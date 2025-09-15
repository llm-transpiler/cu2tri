#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// -------------------------------------- FP32
// -------------------------------------- Swish x: N, y: N y=x*sigmoid(x)
__device__ __forceinline__ float swish(float x) {
  return x / (1.0f + expf(-x));
}

__global__ void swish_f32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = swish(reg_x.x);
    reg_y.y = swish(reg_x.y);
    reg_y.z = swish(reg_x.z);
    reg_y.w = swish(reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

extern "C" void cuda_kernel(float* x, float* y, int N) {
    dim3 block(256 / 4);  // 64 threads
    dim3 grid((N + 256 - 1) / 256);
    swish_f32x4_kernel<<<grid, block>>>(x, y, N);
}
