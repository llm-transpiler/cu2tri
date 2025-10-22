import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel that performs element‑wise addition.
# Mirrors the behavior of the original CUDA kernel (320 threads per block).
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size):
    BLOCK_SIZE = 320                     # matches CUDA blockDim.x
    pid = tl.program_id(0)               # block index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices
    mask = offs < size                    # guard out‑of‑bounds accesses

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper that reproduces the original CUDA launch API.
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point analogous to the original `cuda_kernel` function.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA) – corresponds to `float *A` in CUDA.
    B : torch.Tensor
        Input tensor (float32, CUDA) – corresponds to `float *B` in CUDA.
    C : torch.Tensor
        Output tensor (float32, CUDA) – corresponds to `float *C` in CUDA.
    size : int
        Number of elements to process.
    """
    # ------------------------------------------------------------------
    # Basic sanity checks (mirroring expectations of the CUDA code)
    # ------------------------------------------------------------------
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor), \
        "All inputs must be torch tensors."
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device."
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32."
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor sizes must be at least `size`."

    BLOCK_SIZE = 320
    # Compute grid exactly as the CUDA launch configuration
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel with 10 warps (10 * 32 = 320 threads)
    _triton_kernel_impl[grid](A, B, C, size, num_warps=10)

    # Optional: synchronize to make kernel completion explicit
    torch.cuda.synchronize()