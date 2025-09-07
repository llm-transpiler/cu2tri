#include <assert.h>

__global__ void __launch_bounds__(320)
    kernel(float *__restrict__ A, float *__restrict__ B,
        float *__restrict__ T_add) {
  T_add[((int)threadIdx.x)] = (A[((int)threadIdx.x)] + B[((int)threadIdx.x)]);
}

extern "C" void cuda_kernel(float *A, float *B, float *C, int size) {

  dim3 blockSize(320);
  dim3 numBlocks((size + 320 - 1) / 320);

  kernel<<<numBlocks, blockSize>>>(A, B, C);
}