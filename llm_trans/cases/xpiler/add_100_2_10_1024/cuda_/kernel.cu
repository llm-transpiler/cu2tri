#include <assert.h>

__global__ void __launch_bounds__(1024)
    _cuda_kernel_impl(float *__restrict__ A, float *__restrict__ B,
        float *__restrict__ T_add) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0;
       ax0_ax1_fused_ax2_fused_ax3_fused_outer < 8;
       ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    int index = ((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) +
                 (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x);
    if (index < 2048000) {
      T_add[index] = A[index] + B[index];
    }
  }
}

extern "C" void cuda_kernel(float *A, float *B, float *C, int size) {
  dim3 blockSize(1024);
  dim3 numBlocks(256);
  
  _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, B, C);
}