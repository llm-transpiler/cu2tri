#include <assert.h>

__global__ void __launch_bounds__(1024)
    kernel(float *__restrict__ A, float *__restrict__ pool_avg) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= 256) return;  // 总输出元素数: 1*64*2*2 = 256
  
  // 参数定义
  const int C = 64, H = 5, W = 5;
  const int kernel_size = 3, stride = 2;
  const int output_H = 2, output_W = 2;  // (H-kernel_size)/stride + 1
  
  // 将tid映射到输出坐标(n, c, h, w)
  int n = tid / (C * output_H * output_W);
  int remaining = tid % (C * output_H * output_W);
  int c = remaining / (output_H * output_W);
  remaining = remaining % (output_H * output_W);
  int h = remaining / output_W;
  int w = remaining % output_W;
  
  // 计算sum pooling
  float pool_sum = 0.0f;
  for (int rv0 = 0; rv0 < kernel_size; ++rv0) {
    for (int rv1 = 0; rv1 < kernel_size; ++rv1) {
      int input_h = h * stride + rv0;
      int input_w = w * stride + rv1;
      
      // 边界检查
      if (input_h < H && input_w < W) {
        int input_idx = n * (C * H * W) + c * (H * W) + input_h * W + input_w;
        pool_sum += A[input_idx];
      }
    }
  }
  
  pool_avg[tid] = pool_sum;
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