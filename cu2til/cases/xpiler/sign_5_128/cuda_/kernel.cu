#include <assert.h>

__global__ void __launch_bounds__(640)
    _cuda_kernel_impl(float *__restrict__ A, float *__restrict__ T_sign) {
  T_sign[((int)threadIdx.x)] =
      ((0.000000e+00f < A[((int)threadIdx.x)])
           ? 1.000000e+00f
           : ((A[((int)threadIdx.x)] < 0.000000e+00f) ? -1.000000e+00f
                                                      : 0.000000e+00f));
}

extern "C" void cuda_kernel(float *A, float *C, int size) {
  dim3 blockSize(640);
  dim3 numBlocks((size + 640 - 1) / 640);

  _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, C);
}