#include <assert.h>

__global__ void __launch_bounds__(64)
    _cuda_kernel_impl(float *__restrict__ A, float *__restrict__ B,
        float *__restrict__ T_add) {
  T_add[((int)threadIdx.x)] = (A[((int)threadIdx.x)] + B[((int)threadIdx.x)]);
}

extern "C" void cuda_kernel(float *A, float *B, float *C, int size) {

  dim3 blockSize(64);
  dim3 numBlocks((size + 64 - 1) / 64);

  _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, B, C);
}