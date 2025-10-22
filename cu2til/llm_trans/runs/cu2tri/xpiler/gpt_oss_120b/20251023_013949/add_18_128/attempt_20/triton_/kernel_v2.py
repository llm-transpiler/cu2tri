import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Compile‑time constants (mirroring the CUDA launch bounds and hard‑coded limit)
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024          # threads per block (as in the CUDA kernel)
MAX_ELEMENTS = 2304        # the fixed upper bound used in the original kernel

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr,
                        BLOCK_SIZE: tl.constexpr,
                        MAX_ELEMENTS: tl.constexpr):
    """
    Triton implementation of the element‑wise addition kernel.
    It reproduces the exact semantics of the original CUDA kernel:
        if (global_idx < 2304) {
            T_add[global_idx] = A[global_idx] + B[global_idx];
        }
    """
    pid = tl.program_id(0)                     # block index (gridDim.x)
    block_start = pid * BLOCK_SIZE              # first element handled by this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)   # per‑thread indices

    # Apply the same hard‑coded bound check as the CUDA version
    mask = offsets < MAX_ELEMENTS

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(T_add_ptr + offsets, a + b, mask=mask)


def triton_kernel(A: torch.Tensor,
                  B: torch.Tensor,
                  C: torch.Tensor,
                  size: int):
    """
    Entry‑point that mimics the original ``cuda_kernel`` signature.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA, contiguous).
    B : torch.Tensor
        Input tensor B (float32, CUDA, contiguous).
    C : torch.Tensor
        Output tensor C (float32, CUDA, contiguous).
    size : int
        Number of elements (used only for grid configuration, identical to the CUDA wrapper).
    """
    # ------------------------------------------------------------------
    # Basic validation – mirrors expectations of the original launch code
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"

    # Ensure contiguous layout for correct pointer arithmetic inside the kernel
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # ------------------------------------------------------------------
    # Grid configuration: identical to the CUDA wrapper
    #   numBlocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    # ------------------------------------------------------------------
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # ------------------------------------------------------------------
    # Launch the Triton kernel.
    # BLOCK_SIZE and MAX_ELEMENTS are passed as compile‑time arguments.
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](A, B, C, BLOCK_SIZE, MAX_ELEMENTS)