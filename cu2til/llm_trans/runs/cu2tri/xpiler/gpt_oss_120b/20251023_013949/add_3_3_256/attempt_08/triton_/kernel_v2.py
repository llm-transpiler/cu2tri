import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Triton kernel: elementwise addition with the same hard‑coded bound (2304)
# as the original CUDA implementation.
# -------------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A: tl.pointer(tl.float32),
    B: tl.pointer(tl.float32),
    C: tl.pointer(tl.float32),
    BLOCK_SIZE: tl.constexpr = 1024,
    BOUND: tl.constexpr = 2304,
):
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices
    mask = offsets < BOUND                      # replicate CUDA's if‑condition

    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, a + b, mask=mask)


# -------------------------------------------------------------------------
# Wrapper matching the original CUDA kernel signature.
# -------------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton implementation of the original CUDA kernel.

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D tensors of dtype torch.float32 residing on the same CUDA device.
    size : int
        Logical size of the vectors (used only to compute the grid size,
        identical to the host code of the CUDA version).
    """
    # -----------------------------------------------------------------
    # Basic sanity checks (mirroring expectations of the CUDA launch)
    # -----------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on a CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"

    BLOCK_SIZE = 1024
    # Compute number of blocks exactly as the CUDA host code does.
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel with enough warps to cover BLOCK_SIZE threads.
    # BLOCK_SIZE // 32 = 32 warps (1024 threads) matches the compile‑time BLOCK_SIZE.
    _triton_kernel_impl[grid](A, B, C, num_warps=BLOCK_SIZE // 32)