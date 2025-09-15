#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void embedding_f32x4_pack_kernel(const int *idx, float *weight,
                                           float *output, int n, int emb_size) {
  int tx = threadIdx.x * 4;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  
  // 安全的float4打包读写
  if (tx + 3 < emb_size) {
    float4 pack_weight = *(float4*)&weight[offset + tx];
    *(float4*)&output[bx * emb_size + tx] = pack_weight;
  } else {
    // 边界情况，逐个复制
    for (int i = 0; i < 4 && tx + i < emb_size; i++) {
      output[bx * emb_size + tx + i] = weight[offset + tx + i];
    }
  }
}

extern "C" void cuda_kernel(const int* idx, float* weight, float* output, int n, int emb_size) {
    dim3 block(emb_size / 4);  // emb_size/4 threads
    dim3 grid(n);              // n blocks
    embedding_f32x4_pack_kernel<<<grid, block>>>(idx, weight, output, n, emb_size);
}
