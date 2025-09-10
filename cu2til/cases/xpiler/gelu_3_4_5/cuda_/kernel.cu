#include <assert.h>
#include <cmath>

// Forward declaration of the device function
__device__ float geluf(float x);

__device__ float geluf(float x) {
  return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}

__global__ void __launch_bounds__(60)
    _cuda_kernel_impl(float *__restrict__ A, float *__restrict__ compute) {
  compute[((int)threadIdx.x)] = geluf(A[((int)threadIdx.x)]);
}

extern "C" void cuda_kernel(float *A, float *C, int size) {
  dim3 blockSize(60);
  dim3 numBlocks((size + 60 - 1) / 60);

  _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, C);
}