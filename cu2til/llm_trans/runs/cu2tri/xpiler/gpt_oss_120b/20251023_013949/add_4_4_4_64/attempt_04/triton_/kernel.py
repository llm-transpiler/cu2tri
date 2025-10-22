import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing vector addition (A + B -> C)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A, B, C, N, BLOCK_SIZE: tl.constexpr):
    """
    Parameters
    ----------
    A : pointer to float32
    B : pointer to float32
    C : pointer to float32
    N : int32, number of elements to process
    BLOCK_SIZE : compile‑time constant, threads per block (1024)
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices
    mask = offsets < N                         # guard out‑of‑bounds

    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton entry point that mirrors the CUDA kernel:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Arguments
    ---------
    A : torch.Tensor (float32, CUDA, contiguous)
    B : torch.Tensor (float32, CUDA, contiguous)
    C : torch.Tensor (float32, CUDA, contiguous) – output
    size : int – number of elements to process
    """
    # ------------------- sanity checks -------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must reside on the CUDA device.")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise ValueError("All tensors must be of dtype torch.float32.")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise ValueError("All tensors must be contiguous.")
    if size < 0:
        raise ValueError("size must be non‑negative.")
    if size > A.numel() or size > B.numel() or size > C.numel():
        raise ValueError("size exceeds the length of one or more tensors.")

    # ------------------- launch configuration -------------------
    BLOCK_SIZE = 1024
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )   # 1‑D grid

    # ------------------- kernel launch -------------------
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    # Ensure the kernel has finished before returning
    torch.cuda.synchronize()