#include <assert.h>

__global__ void __launch_bounds__(1024)
    kernel(float *__restrict__ A, float *__restrict__ T_softmax_exp) {
  int idx = blockIdx.x * 1024 + threadIdx.x;
  if (idx < 1380) {

    float maxVal = A[idx * 128];
    for (int i = 1; i < 128; ++i) {
      if (A[idx * 128 + i] > maxVal) {
        maxVal = A[idx * 128 + i];
      }
    }

    float denom = 0.0f;
    for (int i = 0; i < 128; ++i) {
      T_softmax_exp[idx * 128 + i] = expf(A[idx * 128 + i] - maxVal);
      denom += T_softmax_exp[idx * 128 + i];
    }

    for (int i = 0; i < 128; ++i) {
      T_softmax_exp[idx * 128 + i] /= denom;
    }
  }
}

extern "C" void cuda_kernel(float *A, float *C, int size1, int size2) {
  dim3 blockSize(1024);
  dim3 numBlocks((size1 + 1024 - 1) / 1024);

  kernel<<<numBlocks, blockSize>>>(A, C);
}