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

// HGEMV: Warp HGEMV K128
__global__ void hgemv_k128_f16x4_kernel(half *a, half *x, half *y, int M, int K) {
  int tx = threadIdx.x;         // 0~31
  int ty = threadIdx.y;         // 0~4
  int bx = blockIdx.x;          // 0~M/4
  int lane = tx % WARP_SIZE;    // 0~31
  int m = bx * blockDim.y + ty; // (0~M/4) * 4 + (0~3)
  
  if (m < M) {
    half sum = __float2half(0.0f);
    int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
    
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      int k = w * WARP_SIZE + lane;
      if (k < K) {
        sum = __hadd(sum, __hmul(a[m * K + k], x[k]));
      }
    }
    
    // Warp reduce
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
      sum = __hadd(sum, __shfl_xor_sync(0xffffffff, sum, mask));
    }
    
    if (lane == 0)
      y[m] = sum;
  }
}

extern "C" void cuda_kernel(half* a, half* x, half* y, int M, int K) {
    dim3 block(32, 4);  // 32x4 threads per block
    dim3 grid((M + 4 - 1) / 4);  // M/4 blocks
    hgemv_k128_f16x4_kernel<<<grid, block>>>(a, x, y, M, K);
}
