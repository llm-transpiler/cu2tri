import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    tid = tl.thread_id(0)
    for outer in range(8):
        index = outer * 262144 + pid * BLOCK_SIZE + tid
        if index < 2048000:
            a_val = tl.load(A_ptr + index)
            b_val = tl.load(B_ptr + index)
            tl.store(C_ptr + index, a_val + b_val)

def triton_kernel(A, B, C, size):
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be on CUDA device")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise ValueError("All tensors must be float32")
    grid = (256,)
    BLOCK_SIZE = 1024
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)