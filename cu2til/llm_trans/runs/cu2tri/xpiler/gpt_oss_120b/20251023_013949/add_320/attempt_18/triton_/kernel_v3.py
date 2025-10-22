import torch
import triton
import triton.language as tl

# Number of elements processed per program (block). Must be a multiple of 32.
# Choose 512 (16 warps) to satisfy Triton's power‑of‑two warps requirement.
BLOCK_SIZE = 512

@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *Pointer* to input A (float32)
    B_ptr,          # *Pointer* to input B (float32)
    C_ptr,          # *Pointer* to output C (float32)
    size,           # total number of elements to process (runtime)
    BLOCK_SIZE: tl.constexpr,  # compile‑time constant matching threads per program
):
    """
    Triton kernel that performs element‑wise addition:
        C[i] = A[i] + B[i]   for i < size
    The kernel processes BLOCK_SIZE elements per program; excess threads are masked out.
    """
    pid = tl.program_id(0)                     # program (block) index
    block_start = pid * BLOCK_SIZE              # start offset for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # global indices for each thread
    mask = offsets < size                       # guard against out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int) -> None:
    """
    Wrapper that launches the Triton kernel with the same API as the original CUDA kernel.
    """
    # Input validation (mirrors expectations of the original CUDA code)
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("All tensors must be float32.")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise ValueError("All tensors must be contiguous.")
    if A.device != B.device or A.device != C.device:
        raise ValueError("All tensors must reside on the same device.")
    if A.device.type != "cuda":
        raise RuntimeError("Tensors must be on a CUDA device.")
    if size < 0:
        raise ValueError("size must be non‑negative.")
    if size == 0:
        return  # nothing to do

    # Compute grid size: number of programs (blocks) needed
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the kernel.
    # Use 16 warps (num_warps must be a power of two). This yields 512 threads per program,
    # matching BLOCK_SIZE.
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )