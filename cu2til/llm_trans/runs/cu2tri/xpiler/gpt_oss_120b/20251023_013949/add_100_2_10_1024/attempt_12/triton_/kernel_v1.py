import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for elementwise addition: C = A + B.
    Parameters
    ----------
    A_ptr, B_ptr, C_ptr : pointers to float32 tensors (device memory)
    size : total number of elements to process
    BLOCK_SIZE : compile‑time constant, number of elements processed per program
    """
    pid = tl.program_id(0)                     # program (block) index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size                       # out‑of‑bounds guard

    a = tl.load_ptr + offsets mask=mask)    # load A
    b = tl.load(B_ptr + offsets, mask=mask)    # load B
    tl.store(C_ptr + offsets, a + b, mask=mask)  # store result


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel.
    Mirrors the signature of the original CUDA kernel launcher.
    """
    # Basic sanity checks (mirroring __restrict__ semantics)
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32")
    if A.shape != B.shape or A.shape != C.shape:
        raise RuntimeError("Input tensors must have identical shapes")
    if size > A.numel():
        raise RuntimeError(f"Requested size {size} exceeds tensor length {A.numel()}")

    BLOCK_SIZE = 1024  # matches the original CUDA blockDim.x
    # Compute the number of program instances needed
    grid = (triton.cdiv(size, BLOCK_SIZE),)

    # Launch the kernel with a modest number of warps (each thread processes BLOCK_SIZE/num_warps/32 elements)
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,   # 4 warps → 128 threads per program, each handling 8 elements
    )