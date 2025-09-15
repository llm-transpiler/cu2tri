#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// SGEMM: Warp SGEMM K128
__global__ void sgemv_k128f32x4_kernel(float *a, float *x, float *y, int M, int K) {
  int tx = threadIdx.x;         // 0~31
  int ty = threadIdx.y;         // 0~3
  int bx = blockIdx.x;          // 0~M/4
  int lane = tx % WARP_SIZE;    // 0~31
  int m = bx * blockDim.y + ty; // (0~M/4) * 4 + (0~3)
  
  if (m < M && tx < K) {
    float4 reg_a = FLOAT4(a[m * K + tx * 4]);
    float4 reg_x = FLOAT4(x[tx * 4]);
    
    float sum = 0.0f;
    sum += reg_a.x * reg_x.x;
    sum += reg_a.y * reg_x.y;
    sum += reg_a.z * reg_x.z;
    sum += reg_a.w * reg_x.w;
    
    // Warp reduce
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
      sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }
    
    if (lane == 0)
      y[m] = sum;
  }
}

extern "C" void cuda_kernel(float* a, float* x, float* y, int M, int K) {
    dim3 block(32, 4);  // 32x4 threads per block
    dim3 grid((M + 4 - 1) / 4);  // M/4 blocks
    sgemv_k128f32x4_kernel<<<grid, block>>>(a, x, y, M, K);
}
