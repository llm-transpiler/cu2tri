import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N: tl.int32, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)               # program (block) index
    num_blocks = tl.num_programs(0)       # total number of programs (grid size)
    # Base offsets for the first iteration
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    for i in range(8):
        idx = offsets + i * num_blocks * BLOCK_SIZE
        mask = idx < N
        a = tl.load(A_ptr + idx, mask=mask, other=0.0)
        b = tl.load(B_ptr + idx, mask=mask, other=0.0)
        tl.store(C_ptr + idx, a + b, mask=mask)

def triton_kernel(A, B, C, size):
    """
    Triton wrapper mimicking the original CUDA kernel signature.
    Parameters:
        A (torch.Tensor): Input tensor on CUDA device, dtype float32.
        B (torch.Tensor): Input tensor on CUDA device, dtype float32.
        C (torch.Tensor): Output tensor on CUDA device, dtype float32.
        size (int): Number of elements to process.
    """
    # Validate inputs
    assert isinstance(A, torch.Tensor) and A.is_cuda, "A must be a CUDA tensor"
    assert isinstance(B, torch.Tensor) and B.is_cuda, "B must be a CUDA tensor"
    assert isinstance(C, torch.Tensor) and C.is_cuda, "C must be a CUDA tensor"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor sizes must be at least 'size'"

    size = int(size)  # ensure Python int

    BLOCK_SIZE = 1024  # matches original CUDA block size (1024 threads per block)
    # Each program instance processes BLOCK_SIZE * 8 elements
    grid = (triton.cdiv(size, BLOCK_SIZE * 8),)

    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )