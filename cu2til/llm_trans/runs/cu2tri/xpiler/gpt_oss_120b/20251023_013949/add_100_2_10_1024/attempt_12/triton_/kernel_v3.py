import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """Elementwise addition C = A + B."""
    pid = tl.program_id(0)                     # program (block) index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size                       # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel.
    Mirrors the signature of the original CUDA launcher.
    """
    # Sanity checks (mirroring __restrict__ semantics)
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be torch.float32")
    if A.shape != B.shape or A.shape != C.shape:
        raise RuntimeError("All tensors must have identical shapes")
    if size > A.numel():
        raise RuntimeError(f"Requested size {size} exceeds tensor length {A.numel()}")

    BLOCK_SIZE = 1024  # matches original blockDim.x
    grid = (triton.cdiv(size, BLOCK_SIZE),)  # one program per BLOCK_SIZE elements

    # Launch the kernel
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,   # reasonable default for this workload
    )