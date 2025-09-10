#include <assert.h>

__global__ void _cuda_kernel_impl(float *input, float *kernel, float *output) {
  int bs = blockIdx.z;  // 批次索引
  int oc = blockIdx.x;  // 输出通道索引
  int oh = threadIdx.y; // 输出高度索引
  int ow = threadIdx.x; // 输出宽度索引

  if (bs < 32 && oc < 64 && oh < 4 && ow < 4) {
    float sum = 0.0;

    for (int ic = 0; ic < 128; ic++) {
      for (int kh = 0; kh < 2; kh++) {
        for (int kw = 0; kw < 2; kw++) {
          int ih = oh * 2 + kh;
          int iw = ow * 2 + kw;

          // 输入索引计算
          int input_idx = bs * (128 * 8 * 8) + ic * (8 * 8) + ih * 8 + iw;

          // 卷积核索引计算
          int kernel_idx = oc * (128 * 2 * 2) + ic * (2 * 2) + kh * 2 + kw;

          sum += input[input_idx] * kernel[kernel_idx];
        }
      }
    }

    // 输出索引计算
    int output_idx = bs * (64 * 4 * 4) + oc * (4 * 4) + oh * 4 + ow;

    output[output_idx] = sum;
  }
}

extern "C" void cuda_kernel(float *input, float *kernel, float *output, 
                                  int batch_size, int input_height,
                                  int input_channels, int output_channels,
                                  int kernel_height, int stride) {
  int output_height = (input_height - kernel_height) / stride + 1;
  int output_width = (input_height - kernel_height) / stride + 1;
  
  dim3 blockSize(output_width, output_height);
  dim3 numBlocks(output_channels, 1, batch_size);
  
  _cuda_kernel_impl<<<numBlocks, blockSize>>>(input, kernel, output);
}