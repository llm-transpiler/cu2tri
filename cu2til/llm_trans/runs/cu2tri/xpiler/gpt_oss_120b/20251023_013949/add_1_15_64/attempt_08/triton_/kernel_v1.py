import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel that reproduces the exact behavior of the original CUDA kernel.
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *Pointer* to the first input tensor (float32)
    B_ptr,          # *Pointer* to the second input tensor (float32)
    C_ptr,          # *Pointer* to the output tensor (float32)
    BLOCK_SIZE: tl.constexpr  # compile‑time constant = 960
):
    # Thread index inside the block (0 … BLOCK_SIZE‑1)
    offs = tl.arange(0, BLOCK_SIZE)

    # Load, add, and store. No bounds checking – mirrors the CUDA kernel.
    a = tl.load(A_ptr + offs)
    b = tl.load(B_ptr + offs)
    tl.store(C_ptr + offs, a + b)


# ----------------------------------------------------------------------
# Wrapper that matches the original CUDA kernel signature.
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton entry‑point mirroring:
        cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA) of length >= size.
    B : torch.Tensor
        Input tensor (float32, CUDA) of length >= size.
    C : torch.Tensor
        Output tensor (float32, CUDA) of length >= size.
    size : int
        Logical vector length; used only to compute the launch grid.
    """
    # Basic sanity checks
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32.")
    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    BLOCK_SIZE = 960
    # Compute number of Triton program instances (equivalent to CUDA blocks)
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the kernel.
    # Number of warps per program = BLOCK_SIZE / 32 (30 for 960 threads)
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32,
    )