import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel that performs element‑wise addition.
# Mirrors the original CUDA kernel (320 threads per block) while using a
# power‑of‑two number of warps (required by Triton).
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size):
    # Compile‑time constant matching the CUDA block size
    BLOCK_SIZE = 320                     # 10 warps × 32 = 320 threads (conceptually)
    pid = tl.program_id(0)               # block index
    # Global offsets for this block
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Guard against out‑of‑bounds accesses
    mask = offs < size

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
    # Sanity checks (mirroring expectations of the CUDA code)
    # ------------------------------------------------------------------
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor), \
        "All inputs must be torch tensors."
    assert A.is_cuda and B.is_cuda and C_cuda, "All tensors must reside on the CUDA device."
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32."
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor sizes must be at least `size`."

    BLOCK_SIZE = 320
    # Compute grid exactly as the CUDA launch configuration
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Triton requires num_warps to be a power of 2; we use 1 warp (32 threads)
    # The kernel internally processes BLOCK_SIZE elements per block via vectorization.
    _triton_kernel_impl[grid](A, B, C, size, num_warps=1)

    # Ensure kernel completion before returning
    torch.cuda.synchronize()