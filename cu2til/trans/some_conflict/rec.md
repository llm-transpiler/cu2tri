/workspace/cu2til/cases/xpiler/add_100_2_10_1024/cuda_/kernel.cu
```cpp
#include <assert.h>

__global__ void __launch_bounds__(1024)
    _cuda_kernel_impl(float *__restrict__ A, float *__restrict__ B,
        float *__restrict__ T_add) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0;
       ax0_ax1_fused_ax2_fused_ax3_fused_outer < 8;
       ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) +
         ((int)blockIdx.x)) < 2048000) {
      T_add[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) +
              (((int)blockIdx.x) * 1024)) +
             ((int)threadIdx.x))] =
          (A[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) +
               (((int)blockIdx.x) * 1024)) +
              ((int)threadIdx.x))] +
           B[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) +
               (((int)blockIdx.x) * 1024)) +
              ((int)threadIdx.x))]);
    }
  }
}

extern "C" void cuda_kernel(float *A, float *B, float *C, int size) {
  dim3 blockSize(1024);
  dim3 numBlocks(256);
  
  _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, B, C);
}
```
这傻逼kernel越界了
```python
    size = A.numel()  # Total number of elements
    # Create output tensor
    output_cuda = torch.empty_like(A)
    
    # Get GPU pointers
    A_ptr = A.cuda().contiguous().data_ptr()
    B_ptr = B.cuda().contiguous().data_ptr()
    output_ptr = output_cuda.data_ptr()
    
    # Call CUDA kernel
    cuda_kernel(A_ptr, B_ptr, output_ptr, size)
    compare_results(output_torch, output_cuda, atol=1e-4)
```
正好让kernel的输出在torch之后了，但是改了输入构造变为
```python

def get_cuda_torch_inputs(params: Params):
    torch.manual_seed(SEED)
    # Create data directly on specified device
    A = torch.randn(params.shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    B = torch.randn(params.shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    
    # Create output tensor
    output_cuda = torch.empty_like(A)
    cuda_output_tensors = [output_cuda]
    
    cuda_all_inputs = [A, B, output_cuda, params.total_elements]
    
    # For PyTorch, inputs are already in correct format
    torch_all_inputs = [A, B]
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

```
之后，output_torch正好在output_cuda后面一点所以被覆盖了