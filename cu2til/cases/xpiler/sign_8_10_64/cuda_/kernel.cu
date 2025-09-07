#include <assert.h>

__global__ void __launch_bounds__(1024)
    kernel(float *__restrict__ A, float *__restrict__ T_sign) {
  if (((blockIdx.x * 1024) + (threadIdx.x)) < 5120) {
    T_sign[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] =
        ((0.000000e+00f < A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))])
             ? 1.000000e+00f
             : ((A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] <
                 0.000000e+00f)
                    ? -1.000000e+00f
                    : 0.000000e+00f));
  }
}

extern "C" void cuda_kernel(float *A, float *C, int size) {
  dim3 blockSize(1024);
  dim3 numBlocks((size + 1024 - 1) / 1024);

  kernel<<<numBlocks, blockSize>>>(A, C);
}