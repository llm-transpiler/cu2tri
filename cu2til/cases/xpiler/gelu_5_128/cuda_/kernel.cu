#include <assert.h>
#include <cmath>

// Forward declaration of the device function
__device__ float geluf(float x);

__device__ float geluf(float x) {
  return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}

__global__ void __launch_bounds__(640)
    kernel(float *__restrict__ A, float *__restrict__ compute) {
  compute[((int)threadIdx.x)] = geluf(A[((int)threadIdx.x)]);
}

extern "C" void cuda_kernel(float *A, float *C, int size) {
  dim3 blockSize(640);
  dim3 numBlocks((size + 640 - 1) / 640);

  kernel<<<numBlocks, blockSize>>>(A, C);
}