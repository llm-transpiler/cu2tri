import torch
import triton
import triton.language as tl

# Block size matches the CUDA implementation
BLOCK_SIZE = 256

@triton.jit
def _triton_kernel_impl(
    a_ptr,   # const float*
    b_ptr,   # const float*
    out_ptr, # float*
    total,   # int32
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = a - b
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_kernel(a: torch.Tensor, b: torch.Tensor, output: torch.Tensor, total: int):
    """
    Triton wrapper that replicates the behavior of the original CUDA kernel.
    Parameters
    ----------
    a : torch.Tensor
        Input tensor (float32, CUDA) of length >= total.
    b : torch.Tensor
        Input tensor (float32, CUDA) of length >= total.
    output : torch.Tensor
        Output tensor (float32, CUDA) of length >= total.
    total : int
        Number of elements to process.
    """
    # Validate inputs
    assert a.is_cuda and b.is_cuda and output.is_cuda, "All tensors must reside on CUDA"
    assert a.dtype == torch.float32 and b.dtype == torch.float32 and output.dtype == torch.float32, "All tensors must be float32"
    assert a.numel() >= total and b.numel() >= total and output.numel() >= total, "Tensor sizes must be at least `total`"

    # Compute grid dimensions
    grid = (triton.cdiv(total, BLOCK_SIZE), )

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        a_ptr=a,
        b_ptr=b,
        out_ptr=output,
        total=total,
        BLOCK_SIZE=BLOCK_SIZE
    )
    # Optional: synchronize if you need to ensure completion before next CPU operation
    # torch.cuda.synchronize()

# Example usage (can be removed in production)
if __name__ == "__main__":
    total = 1_048_576  # 1M elements
    a = torch.randn(total, dtype=torch.float32, device='cuda')
    b = torch.randn(total, dtype=torch.float32, device='cuda')
    out = torch.empty_like(a)
    triton_kernel(a, b, out, total)
    torch.testing.assert_allclose(out, a - b)
    print("Triton kernel executed correctly.")