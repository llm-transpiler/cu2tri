import torch
import triton
import triton.language as tl

@triton.jit
def binary_pointwise_kernel(X, Y, Out, n, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n

    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    o = x + y
    tl.store(Out + offsets, o, mask=mask)

def triton_kernel(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, n: int) -> torch.Tensor:
    """Triton实现：element-wise加法"""
    # Ensure inputs are contiguous
    x = x.contiguous()
    y = y.contiguous()

    BLOCK_N = 1024
    grid = (triton.cdiv(n, BLOCK_N),)

    binary_pointwise_kernel[grid](
        x, y, output, n, BLOCK_N=BLOCK_N, num_warps=8, num_stages=1
    )
    return output

def binary_add_tensor(x, y):
    """保持向后兼容的函数名"""
    dtype = torch.promote_types(x.dtype, y.dtype)
    x, y = torch.broadcast_tensors(x, y)
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x, dtype=dtype)
    n = out.numel()
    BLOCK_N = 1024
    grid = (triton.cdiv(n, BLOCK_N), 1, 1)
    binary_pointwise_kernel[grid](
        x, y, out, n, BLOCK_N=BLOCK_N, num_warps=8, num_stages=1
    )
    return out