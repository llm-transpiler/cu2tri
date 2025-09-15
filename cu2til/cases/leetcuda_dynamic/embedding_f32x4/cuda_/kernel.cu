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

__global__ void embedding_f32x4_kernel(const int *idx, float *weight,
                                       float *output, int n, int emb_size) {
  int tx = threadIdx.x * 4;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  output[bx * emb_size + tx] = weight[offset + tx];
  output[bx * emb_size + tx + 1] = weight[offset + tx + 1];
  output[bx * emb_size + tx + 2] = weight[offset + tx + 2];
  output[bx * emb_size + tx + 3] = weight[offset + tx + 3];
}

extern "C" void cuda_kernel(const int* idx, float* weight, float* output, int n, int emb_size) {
    dim3 block(emb_size / 4);  // emb_size/4 threads
    dim3 grid(n);              // n blocks
    embedding_f32x4_kernel<<<grid, block>>>(idx, weight, output, n, emb_size);
}
