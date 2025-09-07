#include <assert.h>

__global__ void kernel(float *A, float *x, float *y) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < 112) {
    float sum = 0.0f;
    for (int i = 0; i < 128; i++) {
      sum += A[row * 128 + i] * x[i];
    }
    y[row] = sum;
  }
}

extern "C" void cuda_kernel(float *A, float *x, float *y, int m, int n) {
  dim3 blockSize(112);
  dim3 numBlocks((m + 112 - 1) / 112);
  
  kernel<<<numBlocks, blockSize>>>(A, x, y);
}