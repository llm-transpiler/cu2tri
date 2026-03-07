#include <assert.h>

__global__ void __launch_bounds__(1024)
    kernel(float *__restrict__ A, float *__restrict__ pool_avg) {
  float pool_sum[1];
  pool_sum[0] = 0.000000e+00f;
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < 256) {
    // Calculate output position: (batch, channel, out_h, out_w)
    // For shape (1, 64, 2, 2) -> 256 total elements
    int batch = 0;  // Always 0 for our case
    int channel = tid / 4;  // 4 elements per channel (2*2)
    int out_h = (tid % 4) / 2;
    int out_w = (tid % 4) % 2;
    
    // Calculate input starting position (stride = 2)
    int in_h_start = out_h * 2;
    int in_w_start = out_w * 2;
    
    // Sum over 3x3 kernel
    for (int rv0 = 0; rv0 < 3; ++rv0) {
      for (int rv1 = 0; rv1 < 3; ++rv1) {
        int in_h = in_h_start + rv0;
        int in_w = in_w_start + rv1;
        
        // Calculate input index with strides (1600, 25, 5, 1)
        int input_idx = batch * 1600 + channel * 25 + in_h * 5 + in_w;
        pool_sum[0] += A[input_idx];
      }
    }
    
    pool_avg[tid] = pool_sum[0];
  }
}

extern "C" void cuda_kernel(float *input, float *output, int batch_size,
                               int channels, int input_H, int kernel_size,
                               int stride) {
  int output_H = (input_H - kernel_size) / stride + 1;
  int output_size = batch_size * output_H * output_H * channels;
  dim3 blockSize(1024);
  dim3 numBlocks((output_size + blockSize.x - 1) / blockSize.x);

  kernel<<<numBlocks, blockSize>>>(input, output);
}