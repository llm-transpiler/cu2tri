#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void embedding_f16x8_pack_kernel(const int *idx, half *weight,
                                           half *output, int n, int emb_size) {
  int tx = threadIdx.x * 8;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  
  // 安全的128位打包读写
  if ((tx + 7) < emb_size) {
    // 可以安全使用128位读写
    half pack_weight[8], pack_output[8];
    *(float4*)&pack_weight[0] = *(float4*)&weight[offset + tx];
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      pack_output[i] = pack_weight[i];
    }
    
    *(float4*)&output[bx * emb_size + tx] = *(float4*)&pack_output[0];
  } else {
    // 边界情况，逐个复制
    for (int i = 0; i < 8 && tx + i < emb_size; i++) {
      output[bx * emb_size + tx + i] = weight[offset + tx + i];
    }
  }
}

extern "C" void cuda_kernel(const int* idx, half* weight, half* output, int n, int emb_size) {
    dim3 block(emb_size / 8);  // emb_size/8 threads
    dim3 grid(n);              // n blocks
    embedding_f16x8_pack_kernel<<<grid, block>>>(idx, weight, output, n, emb_size);
}
