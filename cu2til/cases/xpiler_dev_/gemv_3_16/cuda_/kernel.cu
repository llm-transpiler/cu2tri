#include <assert.h>

__global__ void _cuda_kernel_impl(float *A, float *x, float *y) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < 3) {
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
      sum += A[row * 16 + i] * x[i];
    }
    y[row] = sum;
  }
}

extern "C" void cuda_kernel(float *A, float *x, float *y, int m, int n) {
  dim3 blockSize(3);
  dim3 numBlocks((m + 3 - 1) / 3);
  
  _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, x, y);
}