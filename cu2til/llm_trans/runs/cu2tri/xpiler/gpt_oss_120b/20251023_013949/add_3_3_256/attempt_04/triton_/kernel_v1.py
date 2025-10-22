import torch
import triton
import triton.language as tl

# Compile‑time constant matching the CUDA kernel's hard‑coded bound
N = 2304

@triton.jit
def _triton_kernel_impl(
    A,                     # pointer to float32
    B,                     # pointer to float32
    T_add,                 # pointer to float32 (output)
    BLOCK_SIZE: tl.constexpr  # threads per program (must be multiple of 32)
):
    """
    Triton kernel that adds two vectors element‑wise.
    Mirrors the behaviour of the original CUDA kernel, including the
    hard‑coded 2304 element bound.
    """
    pid = tl.program_id(0)                     # program (block) index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices
    mask = offsets < N                         # enforce the 2304 limit

    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(T_add + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that launches the Triton kernel.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA).
    B : torch.Tensor
        Input tensor B (float32, CUDA).
    C : torch.Tensor
        Output tensor C (float32, CUDA).
    size : int
        Number of elements to process (used only for grid calculation,
        exactly like the original CUDA wrapper).
    """
    # Basic sanity checks – mirrors the expectations of the CUDA code
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32"

    BLOCK_SIZE = 1024  # matches __launch_bounds__(1024) in the CUDA kernel

    # Compute grid size exactly as the CUDA wrapper does:
    #   numBlocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel. `num_warps` is tuned for Hopper GPUs;
    # 8 warps (256 threads) provides good occupancy for a simple element‑wise kernel.
    _triton_kernel_impl[grid](
        A, B, C,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8
    )