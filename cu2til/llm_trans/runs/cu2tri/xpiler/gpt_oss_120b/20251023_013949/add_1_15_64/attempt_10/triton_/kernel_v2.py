import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,                     # *float32
    B_ptr,                     # *float32
    C_ptr,                     # *float32
    size,                      # int32/64 scalar
    LOGICAL_BLOCK_SIZE: tl.constexpr,  # 960 (actual logical threads per block)
    BLOCK_SIZE: tl.constexpr           # 1024 (power‑of‑two for tl.arange)
):
    """
    Compute element‑wise addition C = A + B for `size` elements.
    Only the first LOGICAL_BLOCK_SIZE threads of each block are active;
    the remaining (BLOCK_SIZE‑LOGICAL_BLOCK_SIZE) threads are masked out.
    """
    pid = tl.program_id(0)                     # block index
    block_start = pid * LOGICAL_BLOCK_SIZE      # start index of this logical block
    # Thread indices within the physical block (0 .. BLOCK_SIZE‑1)
    thread_idx = tl.arange(0, BLOCK_SIZE)
    # Global offsets for all threads in the physical block
    offsets = block_start + thread_idx

    # Mask: (a) thread belongs to the logical block, (b) offset < size
    mask = (thread_idx < LOGICAL_BLOCK_SIZE) & (offsets < size)

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(C_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper with the same signature as the original CUDA kernel
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that mimics the original CUDA kernel:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D tensors on the CUDA device with dtype torch.float32.
    size : int
        Number of elements to process (should match A.numel()).
    """
    # Basic sanity checks
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must reside on the CUDA device.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError("All tensors must be of type torch.float32.")
    if A.shape != B.shape or A.shape != C.shape:
        raise ValueError("All tensors must have the same shape.")
    if size != A.numel():
        raise ValueError("`size` must equal the number of elements in A (and B, C).")

    LOGICAL_BLOCK_SIZE = 960          # matches __launch_bounds__(960)
    BLOCK_SIZE = 1024                 # power of two required by tl.arange

    # Compute grid: one program (block) per logical block of 960 elements
    grid = ((size + LOGICAL_BLOCK_SIZE - 1) // LOGICAL_BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        LOGICAL_BLOCK_SIZE=LOGICAL_BLOCK_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()


# ----------------------------------------------------------------------
# Example usage (can be removed when integrating)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create test data
    size = 12345
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    # Run the Triton kernel
    triton_kernel(A, B, C, size)

    # Verify correctness
    assert torch.allclose(C, A + B), "Result mismatch!"
    print("Triton kernel executed correctly.")