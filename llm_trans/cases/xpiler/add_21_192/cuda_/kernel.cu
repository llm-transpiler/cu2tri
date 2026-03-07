#include <assert.h>

__global__ void __launch_bounds__(1024)
    _cuda_kernel_impl(float *__restrict__ A, float *__restrict__ B,
        float *__restrict__ T_add) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 4032) {
    T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] =
        (A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] +
         B[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" void cuda_kernel(float *A, float *B, float *C, int size) {
  dim3 blockSize(1024);
  dim3 numBlocks((size + 1024 - 1) / 1024);
  
  _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, B, C);
}