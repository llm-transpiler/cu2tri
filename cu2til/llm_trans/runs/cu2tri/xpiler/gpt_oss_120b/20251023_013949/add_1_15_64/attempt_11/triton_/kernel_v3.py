import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE, BLOCK_SIZE_P2: tl.constexpr):
    """Element‑wise addition kernel (A + B → C) with tail handling."""
    pid = tl.program_id(0)  # block index
    block_start = pid * BLOCK_SIZE
    # Range length must be a power‑of‑two constant; we use 1024 (next power of two ≥ 960)
    idx = tl.arange(0, BLOCK_SIZE_P2)
    # Mask out threads beyond the logical block size (960) and beyond the array size
    mask = (idx < BLOCK_SIZE) & (block_start + idx < size)
    a = tl.load(A_ptr + block_start + idx, mask=mask)
    b = tl.load(B_ptr + block_start + idx, mask=mask)
    tl.store(C_ptr + block_start + idx, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel.
    Mirrors the original CUDA entry point `cuda_kernel`.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32) on CUDA device.
    B : torch.Tensor
        Input tensor (float32) on CUDA device.
    C : torch.Tensor
        Output tensor (float32) on CUDA device.
    size : int
        Number of elements to process.
    """
    # Validate inputs
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be on the CUDA device")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be torch.float32")
    # Ensure contiguous layout
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    BLOCK_SIZE = 960          # logical block size (matches the CUDA kernel)
    BLOCK_SIZE_P2 = 1024      # next power‑of‑two for tl.arange
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Triton requires num_warps to be a power of two; 32 warps = 1024 threads per program
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE,
        BLOCK_SIZE_P2=BLOCK_SIZE_P2,
        num_warps=32,
    )