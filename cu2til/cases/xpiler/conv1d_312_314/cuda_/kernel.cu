#include <assert.h>

__global__ void _cuda_kernel_impl(float *input, float *kernel, float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 312) {
    output[idx] = 0;
    for (int j = 0; j < 3; j++) {
      output[idx] += input[idx + j] * kernel[j];
    }
  }
}

extern "C" void cuda_kernel(float *input, float *kernel, float *output, 
                              int input_size, int output_size) {
  dim3 blockSize(312);
  dim3 numBlocks((output_size + blockSize.x - 1) / blockSize.x);
  
  _cuda_kernel_impl<<<numBlocks, blockSize>>>(input, kernel, output);
}