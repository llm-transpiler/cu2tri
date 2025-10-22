import torchimport triton
import triton.language as tl

# Triton kernel implementation
@triton.jit
def _triton_kernel_impl(A, B, C, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise addition: C[i] = A[i] + B[i] for i in [0, N)
    """
    pid = tl.program_id(0)                     # block index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # thread offsets within the block
    mask = offs < N

    a = tl.load(A + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)
    tl.store(C + offs, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA kernel launch.
    Parameters:
        A (torch.Tensor): input tensor of shape (size,) and dtype torch.float32, on CUDA.
        B (torch.Tensor): input tensor of shape (size,) and dtype torch.float32, on CUDA.
        C (torch.Tensor): output tensor of shape (size,) and dtype torch.float32, on CUDA.
        size (int): number of elements to process.
    """
    # Basic validation
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("All tensors must be torch.float32.")
    if size < 0:
        raise ValueError("size must be non‑negative.")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError("Tensor length is smaller than the specified size.")

    BLOCK_SIZE = 320  # matches the original CUDA launch bounds
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)  # 1‑D grid

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )