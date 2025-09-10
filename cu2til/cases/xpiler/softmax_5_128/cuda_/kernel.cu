#include <assert.h>

__global__ void __launch_bounds__(5)
    _cuda_kernel_impl(float *__restrict__ A, float *__restrict__ T_softmax_norm) {
  if (threadIdx.x < 5) {
    int rowStart = threadIdx.x * 128;

    float maxVal = A[rowStart];
    for (int i = 1; i < 128; ++i) {
      if (A[rowStart + i] > maxVal) {
        maxVal = A[rowStart + i];
      }
    }

    float denom = 0.0f;
    for (int i = 0; i < 128; ++i) {
      T_softmax_norm[rowStart + i] = expf(A[rowStart + i] - maxVal);
      denom += T_softmax_norm[rowStart + i];
    }

    for (int i = 0; i < 128; ++i) {
      T_softmax_norm[rowStart + i] /= denom;
    }
  }
}

extern "C" void cuda_kernel(float *A, float *C, int size1, int size2) {
  dim3 blockSize(5);
  dim3 numBlocks((size1 + 5 - 1) / 5);

  _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, C);
}