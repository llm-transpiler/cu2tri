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

// -------------------------------------- FP32
// -------------------------------------- Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// -------------------------------------- FP16
// -------------------------------------- Warp Reduce Sum: Half
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
  float val_f32 = __half2float(val);
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
  }
  return val_f32;
}

template <const int NUM_THREADS = 256 / 8>
__global__ void dot_prod_f16x8_pack_f32_kernel(half *a, half *b, float *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 8; // 8 half elements per thread
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  // temporary register(memory), .local space in ptx, addressable
  half pack_a[8], pack_b[8];                    // 8x16 bits=128 bits.
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
  LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]); // load 128 bits
  const half z = __float2half(0.0f);

  half prod_f16 = z;
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    half2 v = __hmul2(HALF2(pack_a[i]), HALF2(pack_b[i]));
    prod_f16 += (((idx + i) < N) ? (v.x + v.y) : z);
  }

  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
  // warp leaders store the data to shared memory.
  if (lane == 0)
    reduce_smem[warp] = prod;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
  if (tid == 0)
    atomicAdd(y, prod);
}

extern "C" void cuda_kernel(half* a, half* b, float* y, int N) {    
    dim3 block(256 / 8);  // 32 threads
    dim3 grid((N + 256 - 1) / 256);
    dot_prod_f16x8_pack_f32_kernel<<<grid, block>>>(a, b, y, N);
}
