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
    # 8 warps × 32 threads per warp = 256 threads per program (block)
    pid = tl.program_id(0)                # block index
    block_start = pid * 256
    offs = block_start + tl.arange(0, 256)  # compile‑time constant end argument
    mask = offs < size

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper that reproduces the original CUDA launch API.
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton implementation of the original `cuda_kernel` function.
    Parameter list matches the CUDA signature:
        (float *A, float *B, float *C, int size)
    """
    # --------------------------- sanity checks ---------------------------
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor), \
        "All inputs must be torch tensors."
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device."
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32."
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor sizes must be at least `size`."

    BLOCK_SIZE = 256
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Triton requires `num_warps` to be a power of two; 8 warps → 256 threads.
    _triton_kernel_impl[grid](A, B, C, size, num_warps=8)

    # Ensure kernel completion before returning to the caller.
    torch.cuda.synchronize()