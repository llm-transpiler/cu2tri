#include <assert.h>

__global__ void _cuda_kernel_impl(float *A, float *B) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float eps = 1e-5f;

  if (idx < 8192) {
    // Calculate sum
    float sum = 0.0;
    for (int j = 0; j < 8192; j++) {
      sum += A[idx * 8192 + j] * A[idx * 8192 + j];
    }

    // Calculate mean
    float mean = sum / 8192;

    // Calculate scale
    float scale = 1.0 / sqrt(mean + eps);

    // Normalize and store in B
    for (int j = 0; j < 8192; j++) {
      B[idx * 8192 + j] = A[idx * 8192 + j] * scale;
    }
  }
}

extern "C" void cuda_kernel(float *A, float *B, int size1, int size2) {
  dim3 blockSize(1024);
  dim3 numBlocks((size1 + 1024 - 1) / 1024);
  
  _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, B);
}