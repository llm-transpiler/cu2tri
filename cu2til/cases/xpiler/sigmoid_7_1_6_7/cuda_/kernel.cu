#include <assert.h>

__global__ void __launch_bounds__(294)
    kernel(float *__restrict__ A, float *__restrict__ compute) {
  compute[((int)threadIdx.x)] =
      (1.000000e+00f /
       (1.000000e+00f + __expf((0.000000e+00f - A[((int)threadIdx.x)]))));
}

extern "C" void cuda_kernel(float *A, float *C, int size) {
  dim3 blockSize(294);
  dim3 numBlocks((size + 294 - 1) / 294);

  kernel<<<numBlocks, blockSize>>>(A, C);
}