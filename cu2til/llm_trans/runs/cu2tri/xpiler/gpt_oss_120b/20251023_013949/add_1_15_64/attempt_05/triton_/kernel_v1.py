import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementation
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # *Pointer* to input tensor A (float32)
    B_ptr,               # *Pointer* to input tensor B (float32)
    T_add_ptr,           # *Pointer* to output tensor C (float32)
    size,                # total number of elements to process
    BLOCK_SIZE: tl.constexpr  # compile‑time constant: block size (960)
):
    """
    Element‑wise addition: T_add[i] = A[i] + B[i] for i in [0, size).
    Each program instance (block) processes BLOCK_SIZE consecutive elements.
    """
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices
    mask = offsets < size                       # guard out‑of‑bounds

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(T_add_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Launches the Triton kernel to compute C = A + B for `size` elements.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32.
    size : int
        Number of elements to process.
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device"
    assert A.dtype == B.dtype == C.dtype == torch.float32, "All tensors must be float32"
    assert A.shape[0] >= size and B.shape[0] >= size and C.shape[0] >= size, "Tensor sizes must be >= `size`"

    BLOCK_SIZE = 960  # same as __launch_bounds__(960) in the CUDA code

    # Compute grid dimensions: one block per BLOCK_SIZE elements
    grid = lambda meta: ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the kernel
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Optional: synchronize for correctness in a debugging context
    # torch.cuda.synchronize()