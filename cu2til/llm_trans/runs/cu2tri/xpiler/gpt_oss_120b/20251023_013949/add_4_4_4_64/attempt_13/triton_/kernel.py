import torch
import triton
import triton.language as tl

# Block size matches the original CUDA launch bounds
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton implementation of the element‑wise vector addition.
    Parameters:
        A_ptr, B_ptr, C_ptr: pointers to the input and output buffers (float32)
        size: total number of elements to process
        BLOCK_SIZE: compile‑time constant (must be 1024 to match launch bounds)
    """
    pid = tl.program_id(0)                         # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices for this block
    mask = offsets < size                           # guard against OOB

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)  # load A
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)  # load B
    c = a + b                                         # compute addition
    tl.store(C_ptr + offsets, c, mask=mask)           # write result to C

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    A, B, C must be contiguous float32 tensors on the same CUDA device.
    """
    # Basic validation (mirrors typical CUDA host‑side checks)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Tensors must be of type float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "Tensors must be contiguous"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor sizes must be at least 'size'"

    # Compute grid dimensions (one‑dimensional grid)
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A_ptr=A,
        B_ptr=B,
        C_ptr=C,
        size=size,
        BLOCK_SIZE=BLOCK_SIZE,
    )