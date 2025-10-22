import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: element‑wise addition of two vectors.
# Uses 1024 threads per block (power‑of‑two) and masks out the extra 64
# threads so that the logical block size matches the original CUDA kernel
# (960 threads per block).
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *Pointer* to input tensor A (float32)
    B_ptr,          # *Pointer* to input tensor B (float32)
    C_ptr,          # *Pointer* to output tensor C (float32)
    size,           # total number of elements to process (int64)
    BLOCK_SIZE: tl.constexpr,  # logical threads per block (960)
):
    pid = tl.program_id(0)                     # block index
    block_start = pid * BLOCK_SIZE              # start offset for this block

    # Use a power‑of‑two thread range (1024) to satisfy Triton's arange requirement.
    tid = tl.arange(0, 1024)                    # thread index within block (constexpr)
    offsets = block_start + tid

    # Mask for valid work: within the overall size and within the logical block size (960)
    mask = (offsets < size) & (tid < BLOCK_SIZE)

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton entry‑point mirroring the original CUDA kernel signature:
        cuda_kernel(float *A, float *B, float *C, int size)
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must have dtype torch.float32.")

    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    BLOCK_SIZE = 960
    if size <= 0:
        return  # nothing to do

    # ------------------------------------------------------------------
    # Grid configuration (equivalent to CUDA blocks)
    # ------------------------------------------------------------------
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # ------------------------------------------------------------------
    # Launch the kernel.
    # 32 warps = 1024 threads per block (power of two) satisfies Triton's
    # arange requirement while the mask limits active threads to 960.
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )