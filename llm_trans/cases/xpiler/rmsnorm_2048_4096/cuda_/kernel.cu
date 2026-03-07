#include <assert.h>

__global__ void _cuda_kernel_impl(float *A, float *B) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float eps = 1e-5f;

  if (idx < 2048) {
    // Calculate sum
    float sum = 0.0;
    for (int j = 0; j < 4096; j++) {
      sum += A[idx * 4096 + j] * A[idx * 4096 + j];
    }

    // Calculate mean
    float mean = sum / 4096;

    // Calculate scale
    float scale = 1.0 / sqrt(mean + eps);

    // Normalize and store in B
    for (int j = 0; j < 4096; j++) {
      B[idx * 4096 + j] = A[idx * 4096 + j] * scale;
    }
  }
}

extern "C" void cuda_kernel(float *A, float *B, int size_1, int size_2) {
  dim3 blockSize(1024);
  dim3 numBlocks(256);
  
  _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, B);
}