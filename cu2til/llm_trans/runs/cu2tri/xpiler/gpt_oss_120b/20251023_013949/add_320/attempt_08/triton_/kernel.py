import torch
import triton
import triton.language as tl

# Compile‑time block size matching the original CUDA launch bounds
BLOCK_SIZE = 320

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Block (program) identifier
    pid = tl.program_id(0)
    # Absolute offsets for this block's threads
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Guard against out‑of‑bounds accesses
    mask = offsets < N
    # Load inputs
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    # Write result
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton wrapper reproducing the original CUDA kernel:
    C[i] = A[i] + B[i] for i in [0, size).
    """
    # Sanity checks – mirror CUDA expectations
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"
    assert A.device == B.device == C.device, "All tensors must reside on the same device"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor storage must be at least `size` elements"

    # Grid configuration: one block per BLOCK_SIZE
    grid = ((size + BLOCK_SIZE - 1) // BLOCK,)

 # Launch kernel – 10 warps (320/32) matches the original launch bounds
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=10,
    )