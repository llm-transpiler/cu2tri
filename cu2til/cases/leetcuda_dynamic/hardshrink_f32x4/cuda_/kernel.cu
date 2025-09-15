#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LAMBDA 0.5f  // Default threshold for hardshrink

// HardShrink activation function
__global__ void hardshrink_f32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = (fabsf(reg_x.x) > LAMBDA) ? reg_x.x : 0.0f;
    reg_y.y = (fabsf(reg_x.y) > LAMBDA) ? reg_x.y : 0.0f;
    reg_y.z = (fabsf(reg_x.z) > LAMBDA) ? reg_x.z : 0.0f;
    reg_y.w = (fabsf(reg_x.w) > LAMBDA) ? reg_x.w : 0.0f;
    FLOAT4(y[idx]) = reg_y;
  }
}

extern "C" void cuda_kernel(float* x, float* y, int N) {
    dim3 block(256 / 4);  // 64 threads
    dim3 grid((N + 256 - 1) / 256);
    hardshrink_f32x4_kernel<<<grid, block>>>(x, y, N);
}
