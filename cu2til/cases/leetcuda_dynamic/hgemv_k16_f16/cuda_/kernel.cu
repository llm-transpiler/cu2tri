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

// Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// HGEMV: Warp HGEMV K16 - K固定为16
// 假设K为16 < 32,每个warp负责2行，每行有16个元素
// NUM_THREADS=128, NUM_WARPS=NUM_THREADS/WARP_SIZE;
// NUM_ROWS=NUM_WARPS * ROW_PER_WARP, grid(M/NUM_ROWS), block(32,NUM_WARPS)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
template <const int ROW_PER_WARP = 2>
__global__ void hgemv_k16_f16_kernel(half *A, half *x, half *y, int M, int K) {
  constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP;
  int tx = threadIdx.x;      // 0~31
  int ty = threadIdx.y;      // 0~NUM_WARPS
  int bx = blockIdx.x;       // 0~M/NUM_ROWS (NUM_ROWS=NUM_WARPS * ROW_PER_WARP)
  int lane = tx % WARP_SIZE; // 0~31
  int k = lane % K_WARP_SIZE; // 0~15
  // gloabl row of a: MxK and y:Mx1, blockDim.y=NUM_WARPS
  int m = (blockDim.y * bx + ty) * ROW_PER_WARP + lane / K_WARP_SIZE;
  if (m < M) {
    half sum = A[m * K + k] * x[k];
    sum = warp_reduce_sum_f16<K_WARP_SIZE>(sum);
    // 注意是k == 0，而不是lane == 0
    if (k == 0)
      y[m] = sum;
  }
}

extern "C" void cuda_kernel(half* a, half* x, half* y, int M, int K) {
    constexpr int ROW_PER_WARP = 2;
    constexpr int NUM_THREADS = 128;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    constexpr int NUM_ROWS = NUM_WARPS * ROW_PER_WARP;
    dim3 block(32, NUM_WARPS);  // 32x4 threads per block = 128 threads
    dim3 grid((M + NUM_ROWS - 1) / NUM_ROWS);  // M/NUM_ROWS blocks
    hgemv_k16_f16_kernel<ROW_PER_WARP><<<grid, block>>>(a, x, y, M, K);
}
