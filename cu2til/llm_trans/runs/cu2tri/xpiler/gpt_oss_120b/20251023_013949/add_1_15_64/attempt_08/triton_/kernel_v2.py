import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing element‑wise addition.
# Processes up to BLOCK_SIZE elements per program (block) and
# safely handles arbitrary vector lengths via masking.
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *Pointer* to input tensor A (float32)
    B_ptr,          # *Pointer* to input tensor B (float32)
    C_ptr,          # *Pointer* to output tensor C (float32)
    size,           # total number of elements to process (int64)
    BLOCK_SIZE: tl.constexpr,  # compile‑time constant: threads per block (960)
):
    # Program (block) identifier
    pid = tl.program_id(0)
    # Compute the absolute offset for this block
    block_start = pid * BLOCK_SIZE
    # Offsets for the elements this block will handle
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to avoid out‑of‑bounds accesses
    mask = offsets < size
    # Load with mask; out‑of‑bounds lanes receive a dummy value (0.0)
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    # Store the result, respecting the mask
    tl.store(C_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton entry‑point mirroring the original CUDA kernel signature:
        cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32, CUDA) with length >= size.
    B : torch.Tensor
        Input tensor (float32, CUDA) with length >= size.
    C : torch.Tensor
        Output tensor (float32, CUDA) with length >= size.
    size : int
        Logical length of the vectors to be added.
    """
    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is and C.is_cuda):
        raise RuntimeError("All tensors must be tensors.")
    if A.dtype != torch32 or B.dtype != torch.float32 or.dtype != torch.float32:
        raise RuntimeError("All tensors must be of type torch.float32.")
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
    # Launch the Triton kernel.
    # num_warps must be a power of two; 32 warps = 1024 threads ≥ BLOCK_SIZE.
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )