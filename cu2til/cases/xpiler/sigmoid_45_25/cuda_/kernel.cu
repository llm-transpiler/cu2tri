#include <assert.h>

__global__ void __launch_bounds__(1024)
    _cuda_kernel_impl(float *__restrict__ A, float *__restrict__ compute) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 1125) {
    compute[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] =
        (1.000000e+00f /
         (1.000000e+00f +
          __expf((0.000000e+00f -
                  A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]))));
  }
}

extern "C" void cuda_kernel(float *A, float *C, int size) {
  dim3 blockSize(1024);
  dim3 numBlocks((size + 1024 - 1) / 1024);

  _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, C);
}