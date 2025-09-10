#include <assert.h>

__global__ void __launch_bounds__(1024)
    _cuda_kernel_impl(float *__restrict__ A, float *__restrict__ pool_max) {
  float pool_max_local[1];
  pool_max_local[0] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 5; ++rv0) {
    for (int rv1 = 0; rv1 < 5; ++rv1) {
      pool_max_local[0] = max(
          pool_max_local[0],
          A[(((((((((int)blockIdx.x) / 81) * 802816) +
                 (((((((int)blockIdx.x) % 81) * 4) +
                    (((int)threadIdx.x) >> 8)) /
                   9) *
                  21504)) +
                (rv0 * 7168)) +
               ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) % 36) *
                192)) +
              (rv1 * 64)) +
             (((int)threadIdx.x) & 63))]);
    }
  }
  pool_max[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] =
      pool_max_local[0];
}

extern "C" void cuda_kernel(float *input, float *output, int batch_size,
                               int channels, int input_H, int kernel_size,
                               int stride) {
  int output_H = (input_H - kernel_size) / stride + 1;
  int output_size = batch_size * output_H * output_H * channels;
  dim3 blockSize(1024);
  dim3 numBlocks((output_size + blockSize.x - 1) / blockSize.x);

  _cuda_kernel_impl<<<numBlocks, blockSize>>>(input, output);
}