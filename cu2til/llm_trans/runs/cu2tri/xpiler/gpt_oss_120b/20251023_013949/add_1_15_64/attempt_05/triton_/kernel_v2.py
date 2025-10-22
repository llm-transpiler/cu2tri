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
    size,                # Number of elements to process (effective size)
    BLOCK_SIZE: tl.constexpr  # Compile‑time constant: block size (960)
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
        Input tensor of shape (>=size,) and dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of shape (>=size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of shape (>=size,) and dtype torch.float32 on CUDA.
    size : int
        Number of elements to process.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be CUDA tensors")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise ValueError("All tensors must be torch.float32")
    # Compute the effective size that fits within the allocated tensors.
    effective_size = min(int(size), A.shape[0], B.shape[0], C.shape[0])
    if effective_size <= 0:
        return  # nothing to do

    BLOCK_SIZE =  # matches __launch_bounds__(960) in the original CUDA code

    # ------------------------------------------------------------------
    # Grid configuration: one block per BLOCK_SIZE elements
    # ------------------------------------------------------------------
    grid = lambda meta: ((effective_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        effective_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Optional synchronization (uncomment for debugging)
    # torch.cuda.synchronize()