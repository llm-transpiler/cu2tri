#include <assert.h>

__global__ void kernel(const float *input, const float *filter,
                                      float *output) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z; // 使用 blockIdx.z 处理每个通道的独立计算

  // 检查线程是否在输出范围内
  if (tid_x < 4 && tid_y < 4 && c < 128) {
    // 初始化输出值
    float sum = 0.0;

    // 进行 3x3 滤波器的卷积计算
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        int input_x = tid_x + j;
        int input_y = tid_y + i;
        int input_idx = (input_y * 6 + input_x) * 128 + c; // 3 表示通道数
        int filter_idx = (i * 3 + j) * 128 + c;
        sum += input[input_idx] * filter[filter_idx];
      }
    }

    // 将结果存储到输出
    int output_idx = (tid_y * 4 + tid_x) * 128 + c;
    output[output_idx] = sum;
  }
}

extern "C" void cuda_kernel(float *input, float *kernel, float *output,
                                     int input_height, int kernel_size,
                                     int input_channels) {
  dim3 blockSize(256);
  dim3 numBlocks(1);  // 需要根据具体参数调整
  
  kernel<<<numBlocks, blockSize>>>(input, weight, output);
}