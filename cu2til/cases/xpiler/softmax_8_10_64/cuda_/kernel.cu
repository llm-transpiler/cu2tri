#include <assert.h>

__global__ void __launch_bounds__(80)
    _cuda_kernel_impl(float *__restrict__ A, float *__restrict__ T_softmax_exp) {
  if (threadIdx.x < 80) {

    float maxVal = A[threadIdx.x * 64];
    for (int i = 1; i < 64; ++i) {
      if (A[threadIdx.x * 64 + i] > maxVal) {
        maxVal = A[threadIdx.x * 64 + i];
      }
    }

    float denom = 0.0f;
    for (int i = 0; i < 64; ++i) {
      T_softmax_exp[threadIdx.x * 64 + i] =
          expf(A[threadIdx.x * 64 + i] - maxVal);
      denom += T_softmax_exp[threadIdx.x * 64 + i];
    }

    for (int i = 0; i < 64; ++i) {
      T_softmax_exp[threadIdx.x * 64 + i] /= denom;
    }
  }
}

extern "C" void cuda_kernel(float *A, float *C, int size1, int size2) {
  dim3 blockSize(80);
  dim3 numBlocks((size1 + 80 - 1) / 80);

  _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, C);
}