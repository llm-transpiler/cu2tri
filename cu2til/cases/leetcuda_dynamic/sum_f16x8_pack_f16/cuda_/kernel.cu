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

// -------------------------------------- FP16
// -------------------------------------- Warp Reduce Sum: Half
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template <const int NUM_THREADS = 256 / 8>
__global__ void sum_f16x8_pack_f16_kernel(half *a, float *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 8; // 8 half elements per thread
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  // temporary register(memory), .local space in ptx, addressable
  half pack_a[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
  const half z = __float2half(0.0f);

  half sum_f16 = z;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sum_f16 += (((idx + i) < N) ? pack_a[i] : z);
  }

  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce
  // with fp32 inter warps.
  if (lane == 0)
    reduce_smem[warp] = __half2float(sum_f16);
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

extern "C" void cuda_kernel(half* a, float* y, int N) {
    // Clear output
    cudaMemset(y, 0, sizeof(float));
    
    dim3 block(256 / 8);  // 32 threads
    dim3 grid((N + 256 - 1) / 256);
    sum_f16x8_pack_f16_kernel<<<grid, block>>>(a, y, N);
}