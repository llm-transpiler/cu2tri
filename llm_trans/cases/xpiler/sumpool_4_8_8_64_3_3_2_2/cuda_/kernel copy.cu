#include <assert.h>

__global__ void __launch_bounds__(1024)
    kernel(float *__restrict__ A, float *__restrict__ pool_avg) {
  float pool_sum[1];
  pool_sum[0] = 0.000000e+00f;
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < 2304) {
    // NHWC format: input (4, 8, 8, 64), output (4, 3, 3, 64)
    // Calculate output position in NHWC format
    int batch = tid / 576;        // 576 = 3*3*64 elements per batch
    int remainder = tid % 576;
    int out_h = remainder / 192;  // 192 = 3*64 elements per output height
    remainder = remainder % 192;
    int out_w = remainder / 64;   // 64 = 64 elements per output width
    int channel = remainder % 64; // channel index
    
    // Calculate input starting position (stride = 2)
    int in_h_start = out_h * 2;
    int in_w_start = out_w * 2;
    
    // Sum over 3x3 kernel
    for (int rv0 = 0; rv0 < 3; ++rv0) {
      for (int rv1 = 0; rv1 < 3; ++rv1) {
        int in_h = in_h_start + rv0;
        int in_w = in_w_start + rv1;
        
        // Calculate input index in NHWC format: (batch, h, w, c)
        // Input strides: batch=4096, h=512, w=64, c=1
        int input_idx = batch * 4096 + in_h * 512 + in_w * 64 + channel;
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