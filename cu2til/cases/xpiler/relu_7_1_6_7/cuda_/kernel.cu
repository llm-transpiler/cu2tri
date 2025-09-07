#include <assert.h>

__global__ void __launch_bounds__(294)
    kernel(float *__restrict__ A, float *__restrict__ compute) {
  compute[((int)threadIdx.x)] = max(A[((int)threadIdx.x)], 0.000000e+00f);
}

extern "C" void cuda_kernel(float *A, float *C, int size) {
  dim3 blockSize(294);
  dim3 numBlocks((size + 294 - 1) / 294);

  kernel<<<numBlocks, blockSize>>>(A, C);
}