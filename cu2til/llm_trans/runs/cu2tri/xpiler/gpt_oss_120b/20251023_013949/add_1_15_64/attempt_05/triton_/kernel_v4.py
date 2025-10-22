import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementation
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *Pointer* to input tensor A (float32)
    B_ptr,          # *Pointer* to input tensor B (float32)
    T_add_ptr,      # *Pointer* to output tensor (float32)
    size,           # Number of elements to process
    BLOCK_SIZE: tl.constexpr  # Compile‑time constant: block size (960)
):
    """
    Compute T_add[i] = A[i] + B[i] for i in [0, size).
    Each program instance processes BLOCK_SIZE consecutive elements.
    """
    pid = tl.program_id(0)                                 # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices
    mask = offsets < size                                   # out‑of‑bounds guard

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(T_add_ptr + offsets, a + b, mask=mask)


# ----------------------------------------------------------------------
# Wrapper matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Launches the Triton kernel to compute C = A + B for `size` elements.
    Parameters
    ----------
    A : torch.Tensor
        Input tensor (CUDA, float32) with at least `size` elements.
    B : torch.Tensor
        Input tensor (CUDA, float32) with at least `size` elements.
    C : torch.Tensor
        Output tensor (CUDA, float32) with at least `size` elements.
    size : int
        Number of elements to process.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise RuntimeError("All tensors must have dtype torch.float32")

    # Convert possible torch scalar to Python int
    if isinstance(size, torch.Tensor):
        size = int(size.item())
    else:
        size = int(size)

    # Clip to the smallest tensor length to avoid OOB accesses
    effective_size = min(size, A.shape[0], B.shape[0], C.shape[0])
    if effective_size <= 0:
        return  # nothing to do

    BLOCK_SIZE = 960  # matches __launch_bounds__(960) in the original CUDA code

    # ------------------------------------------------------------------
    # Grid configuration: one block per BLOCK_SIZE elements
    # ------------------------------------------------------------------
    grid = ((effective_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

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
    # Optional synchronization for debugging
    # torch.cuda.synchronize()