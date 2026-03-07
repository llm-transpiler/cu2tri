#include <assert.h>

__global__ void __launch_bounds__(60)
    _cuda_kernel_impl(float *__restrict__ A, float *__restrict__ compute) {
  compute[((int)threadIdx.x)] = max(A[((int)threadIdx.x)], 0.000000e+00f);
}

extern "C" void cuda_kernel(float *A, float *C, int size) {
  dim3 blockSize(60);
  dim3 numBlocks((size + 60 - 1) / 60);

  _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, C);
}