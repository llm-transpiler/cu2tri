import torch
import triton
import triton.language as tl

# Triton kernel implementing the same computation as the CUDA kernel.
# Each program (block) processes a fixed chunk of 960 elements.
# The kernel adds the corresponding elements from A and B and writes the result to T_add.
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK_SIZE: tl.constexpr):
    # Thread offsets within the block: 0 .. BLOCK_SIZE-1
    offs = tl.arange(0, BLOCK_SIZE)
    # Mask to avoid out‑of‑bounds accesses when size < BLOCK_SIZE.
    mask = offs < size

    # Load values from A and B (masked)
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # Store the sum into T_add (masked)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA kernel launch.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32.
    size : int
        Number of elements; used to compute grid configuration.
    """
    # Validate inputs
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("All tensors must have dtype torch.float32.")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise RuntimeError("All tensors must be contiguous.")

    BLOCK_SIZE = 960  # Matches __launch_bounds__(960) in the CUDA code

    # Compute grid size: number of programs (blocks) needed to cover `size`.
    # Mirrors the CUDA launch configuration:
    #   dim3 blockSize(960);
    #   dim3 numBlocks((size + 960 - 1) / 960);
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel.
    _triton_kernel_impl[(num_blocks,)](
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )