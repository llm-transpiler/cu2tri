import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < 2304  # replicate original static bound
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A, B, C, size):
    """
    Triton wrapper replicating the original CUDA kernel behavior.
    Parameters:
        A, B, C: torch.Tensor of dtype torch.float32 on CUDA device.
        size: int, total number of elements (used only for grid sizing).
    """
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32
    BLOCK_SIZE = 1024
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    _triton_kernel_impl[(num_blocks,)](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

# Example usage
if __name__ == "__main__":
    size = 2304  # matches the static bound in the kernel
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)
    triton_kernel(A, B, C, size)
    torch.cuda.synchronize()
    # Verify correctness
    if torch.allclose(C, A + B):
        print("Success: Triton kernel matches reference.")
    else:
        print("Mismatch detected.")