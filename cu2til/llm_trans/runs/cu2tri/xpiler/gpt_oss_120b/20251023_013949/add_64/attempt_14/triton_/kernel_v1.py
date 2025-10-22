import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: elementwise addition (A + B -> T_add)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # float* __restrict__ A
    B_ptr,          # float* __restrict__ B
    T_add_ptr,      # float* __restrict__ T_add
    size,           # total number of elements to process
    BLOCK_SIZE: tl.constexpr,  # compile‑time constant (64)
):
    # Program ID identifies the current block (grid dimension 0)
    pid = tl.program_id(0)

    # Compute absolute offsets for the threads in this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Guard against out‑of‑bounds accesses
    mask = offsets < size

    # Load A and B (masked). Use 0.0 as a dummy value for masked lanes.
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    # Store the result (masked)
    tl.store(T_add_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int) -> None:
    """
    Launches the Triton kernel that computes C[i] = A[i] + B[i] for i in [0, size).

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32 on CUDA.
    size : int
        Number of elements to process. Must be <= A.shape[0], B.shape[0], C.shape[0].
    """
    # Basic sanity checks (mirrors the expectations of the CUDA version)
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device."
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32."
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous."

    # Triton launch configuration
    BLOCK_SIZE = 64  # matches __launch_bounds__(64) in the CUDA kernel
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the kernel
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Optional: synchronize for deterministic behavior (not required for correctness)
    torch.cuda.synchronize()


# ----------------------------------------------------------------------
# Example usage (can be removed when integrating into a larger codebase)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Problem size
    N = 1024

    # Allocate input and output tensors on the GPU
    a = torch.randn(N, dtype=torch.float32, device="cuda")
    b = torch.randn(N, dtype=torch.float32, device="cuda")
    c = torch.empty_like(a)

    # Run the Triton kernel
    triton_kernel(a, b, c, N)

    # Verify correctness against PyTorch
    torch.testing.assert_allclose(c, a + b)
    print("Triton kernel executed successfully and results are correct.")