#include <assert.h>

__global__ void depthwiseconv_kernel(const float *input, const float *filter,
                                      float *output) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z; // 使用 blockIdx.z 处理每个通道的独立计算

  // 检查线程是否在输出范围内
  if (tid_x < 254 && tid_y < 254 && c < 3) {
    // 初始化输出值
    float sum = 0.0;

    // 进行 3x3 滤波器的卷积计算
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        int input_x = tid_x + j;
        int input_y = tid_y + i;
        int input_idx = (input_y * 256 + input_x) * 3 + c; // 3 表示通道数
        int filter_idx = (i * 3 + j) * 3 + c;
        sum += input[input_idx] * filter[filter_idx];
      }
    }

    // 将结果存储到输出
    int output_idx = (tid_y * 254 + tid_x) * 3 + c;
    output[output_idx] = sum;
  }
}

extern "C" void cuda_kernel(float *input, float *kernel, float *output,
                                     int input_height, int kernel_size,
                                     int input_channels) {
  int output_height = input_height - kernel_size + 1;
  int output_width = input_height - kernel_size + 1;
  
  // 定义块和网格尺寸
  dim3 blockSize(32, 32);
  dim3 numBlocks((output_width + blockSize.x - 1) / blockSize.x,
                 (output_height + blockSize.y - 1) / blockSize.y,
                 input_channels); // 每个通道使用一个块
  
  depthwiseconv_kernel<<<numBlocks, blockSize>>>(input, kernel, output);
}