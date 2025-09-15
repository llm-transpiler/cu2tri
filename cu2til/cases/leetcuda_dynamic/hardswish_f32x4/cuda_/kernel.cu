#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// HardSwish activation function: x * ReLU6(x + 3) / 6
__global__ void hardswish_f32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    
    // HardSwish: x * ReLU6(x + 3) / 6
    float temp_x = reg_x.x + 3.0f;
    reg_y.x = reg_x.x * fminf(fmaxf(temp_x, 0.0f), 6.0f) / 6.0f;
    
    temp_x = reg_x.y + 3.0f;
    reg_y.y = reg_x.y * fminf(fmaxf(temp_x, 0.0f), 6.0f) / 6.0f;
    
    temp_x = reg_x.z + 3.0f;
    reg_y.z = reg_x.z * fminf(fmaxf(temp_x, 0.0f), 6.0f) / 6.0f;
    
    temp_x = reg_x.w + 3.0f;
    reg_y.w = reg_x.w * fminf(fmaxf(temp_x, 0.0f), 6.0f) / 6.0f;
    
    FLOAT4(y[idx]) = reg_y;
  }
}

extern "C" void cuda_kernel(float* x, float* y, int N) {
    dim3 block(256 / 4);  // 64 threads
    dim3 grid((N + 256 - 1) / 256);
    hardswish_f32x4_kernel<<<grid, block>>>(x, y, N);
}
