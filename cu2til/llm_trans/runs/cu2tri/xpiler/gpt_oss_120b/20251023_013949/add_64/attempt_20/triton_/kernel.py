import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, BLOCK_SIZE: tl.constexpr):
    # Block (grid) index – present for compatibility but not used,
    # mirroring the original CUDA kernel which only used threadIdx.x.
    pid = tl.program_id(0)
    # Thread index within the block (0 .. BLOCK_SIZE-1)
    tid = tl.arange(0, BLOCK_SIZE)
    # Load values from the input tensors
    a = tl.load(A_ptr + tid)
    b = tl.load(B_ptr + tid)
    # Compute the element‑wise sum
    c = a + b
    # Store the result back to the output tensor
    tl.store(T_add_ptr + tid, c)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton wrapper that reproduces the semantics of the original CUDA kernel.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32, residing on CUDA.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32, residing on CUDA.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32, residing on CUDA.
    size : int
        Number of elements (used only to compute the launch grid size).
    """
    # Basic sanity checks – the original CUDA code expects raw device pointers.
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    BLOCK_SIZE = 64
    # Compute the number of blocks exactly as in the CUDA launch configuration.
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the Triton kernel. The grid dimension matches the CUDA grid.
    _triton_kernel_impl[(num_blocks,)](A, B, C, BLOCK_SIZE)