import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing the exact behavior of the original CUDA kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Each Triton program (i.e., block) processes exactly BLOCK_SIZE threads.
    The computation mirrors the CUDA kernel:
        T_add[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x]
    No blockIdx offset is applied, so every block writes to the same
    indices 0 .. BLOCK_SIZE-1.
    """
    # Thread indices within the block (0 .. BLOCK_SIZE-1)
    offs = tl.arange(0, BLOCK_SIZE)

    # Load values from global memory
    a = tl.load(A_ptr + offs)
    b = tl.load(B_ptr + offs)

    # Perform element‑wise addition
    c = a + b

    # Store the result back to global memory
    tl.store(T_add_ptr + offs, c)


# ----------------------------------------------------------------------
# Wrapper function matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton entry point that reproduces the semantics of the CUDA launch.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor (float32) residing on the CUDA device.
    B : torch.Tensor
        Input tensor (float32) residing on the CUDA device.
    C : torch.Tensor
        Output tensor (float32) residing on the CUDA device.
    size : int
        Number of elements (used only for grid sizing, exactly as in the
        original CUDA code).
    """
    # ------------------------------------------------------------------
    # Sanity checks (mirroring the expectations of the original CUDA code)
    # ------------------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of dtype torch.float32"

    # Ensure contiguous memory layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # ------------------------------------------------------------------
    # Grid configuration – identical to the CUDA launch parameters
    # ------------------------------------------------------------------
    BLOCK_SIZE = 320  # matches __launch_bounds__(320) in the CUDA kernel
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE, )  # 1‑D grid

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=10,          # 320 threads = 10 warps (32 threads per warp)
    )