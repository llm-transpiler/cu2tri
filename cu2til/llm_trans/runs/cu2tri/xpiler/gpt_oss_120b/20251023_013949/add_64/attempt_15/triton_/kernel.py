import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, B, T_add, BLOCK_SIZE: tl.constexpr):
    # lane index within the block (0 .. BLOCK_SIZE-1)
    lane = tl.arange(0, BLOCK_SIZE)
    # Offset mirrors the CUDA kernel: only threadIdx.x is used,
    # blockIdx.x is ignored, so every block works on the same indices.
    offset = lane
    # Load the two input elements
    a = tl.load(A + offset)
    b = tl.load(B + offset)
    # Store the element‑wise sum
    tl.store(T_add + offset, a + b)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton equivalent of the CUDA kernel `cuda_kernel`.
    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D float32 tensors on the CUDA device.
    size : int
        Logical vector length (used only for grid sizing, exactly as in the CUDA launch).
    """
    # Validate inputs
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    BLOCK_SIZE = 64
    # Compute grid dimensions exactly like the CUDA code
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the Triton kernel. Each block processes only the first BLOCK_SIZE elements,
    # reproducing the original CUDA kernel's behavior (including the intentional data race).
    _triton_kernel_impl[(num_blocks,)](A, B, C, BLOCK_SIZE=BLOCK_SIZE)