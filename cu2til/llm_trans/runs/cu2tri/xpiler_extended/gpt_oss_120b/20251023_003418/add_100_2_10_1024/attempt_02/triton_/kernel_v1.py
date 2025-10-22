import
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr,, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes C = A + B element‑wise for N elements.
    """
    pid = tl.program_id(0)                     # 1‑D grid of program IDs
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mirrors the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D tensors of dtype torch.float32 located on the CUDA device.
    size : int
        Number of elements to process (must be ≤ A.numel()).
    """
    # Validate inputs
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # Block size – 1024 matches the original CUDA blockDim.x
    BLOCK_SIZE = 1024

    # Grid configuration: one‑dimensional launch
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

    # Results are written in‑place to C; no explicit return needed.