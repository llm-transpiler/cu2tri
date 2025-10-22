import torch
import triton
import triton.language as tl

# Triton kernel that mirrors the original CUDA kernel.
# Uses a power‑of‑two tile size (1024) for tl.arange and masks out the extra threads.
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    # Thread offsets within the program (must be a power of two).
    offs = tl.arange(0, 1024)          # 1024 >= BLOCK_SIZE (960)
    pid = tl.program_id(0)            # Block index

    # Global offset for each thread.
    global_offs = pid * BLOCK_SIZE + offs

    # Mask out threads beyond BLOCK_SIZE and beyond the actual size.
    mask = (offs < BLOCK_SIZE) & (global_offs < size)

    a = tl.load(A_ptr + global_offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + global_offs, mask=mask, other=0.0)
    tl.store(C_ptr + global_offs, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA kernel launch.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype.float32 on.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32 on CUDA.
    size : int
        Number of elements to process.
    """
    # Input validation
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("All tensors must have dtype torch.float32.")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise RuntimeError("All tensors must be contiguous.")

    BLOCK_SIZE = 960  # matches __launch_bounds__(960) in the original CUDA code

    # Compute number of program instances (blocks) needed to cover `size`.
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel.
    _triton_kernel_impl[(num_blocks,)](
        A,          # Triton automatically treats torch tensors as pointers
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )