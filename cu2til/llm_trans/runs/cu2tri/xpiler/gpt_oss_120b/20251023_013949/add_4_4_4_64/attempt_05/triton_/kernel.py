import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    A_ptr,               # *Pointer* to the first input array (float32)
    B_ptr,               # *Pointer* to the second input array (float32)
    T_add_ptr,           # *Pointer* to the output array (float32)
    BLOCK_SIZE: tl.constexpr  # Compile‑time constant: number of threads per program
):
    pid = tl.program_id(0)                     # Block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Guard against out‑of‑range accesses – same condition as the CUDA kernel.
    mask = offsets < 4096

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(T_add_ptr + offsets, c, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point matching the original CUDA launch signature:
        void cuda_kernel(float *A, float *B, float *C, int size)
    """
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), "All tensors must be contiguous"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "All tensors must be float32"

    BLOCK_SIZE = 1024
    # Compute grid size exactly as the CUDA host code.
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel.
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()