import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel that mirrors the CUDA implementation
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr):
    """
    Each Triton program corresponds to one CUDA block.
    The program uses 320 threads (10 warps) and each thread works on
    the element indexed by its threadIdx.x, exactly like the CUDA kernel.
    """
    BLOCK_SIZE = 320                     # equivalent to CUDA blockDim.x
    offs = tl.arange(0, BLOCK_SIZE)      # threadIdx.x within the block

    # Load the two input values
    a = tl.load(A_ptr + offs)            # float32 by default
    b = tl.load(B_ptr + offs)

    # Store the element‑wise sum
    tl.store(T_add_ptr + offs, a + b)


# ----------------------------------------------------------------------
# Wrapper that mimics the original CUDA launch API
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that reproduces the behavior of the original `cuda_kernel`
    function.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA) – corresponds to `float *A` in CUDA.
    B : torch.Tensor
        Input tensor (float32, CUDA) – corresponds to `float *B` in CUDA.
    C : torch.Tensor
        Output tensor (float32, CUDA) – corresponds to `float *C` in CUDA.
    size : int
        Number of elements (unused inside the kernel, kept for API compatibility).
    """
    # ------------------------------------------------------------------
    # Basic sanity checks (mirroring the expectations of the CUDA code)
    # ------------------------------------------------------------------
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device."
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32."

    BLOCK_SIZE = 320
    # Compute grid exactly as the CUDA launch configuration
    num_blocks = (size + BLOCK - ) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel.
    # 10 warps = 320 threads per program, matching the CUDA launch bounds.
    _triton_kernel_impl[grid](A, B, C, num_warps=10)