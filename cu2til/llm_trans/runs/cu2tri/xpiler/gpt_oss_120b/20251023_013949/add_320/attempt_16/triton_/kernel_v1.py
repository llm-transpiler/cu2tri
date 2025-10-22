import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise addition.
# The kernel name must be exactly `_triton_kernel_impl`.
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # Pointer to input tensor A (float32)
    B_ptr,          # Pointer to input tensor B (float32)
    T_add_ptr,      # Pointer to output tensor C (float32)
    size,           # Number of elements to process (int)
    BLOCK_SIZE: tl.constexpr  # Compile‑time constant: threads per program
):
    """
    Each program (i.e. block) processes BLOCK_SIZE elements.
    Offsets are computed as:
        offset = pid * BLOCK_SIZE + lane_id
    where `pid` is the program id (grid index) and `lane_id` is the
    thread index inside the block.
    """
    pid = tl.program_id(0)                     # Grid index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size                       # Guard against OOB

    a = tl.load(A_ptr + offsets, mask=mask)    # Load A
    b = tl.load(B_ptr + offsets, mask=mask)    # Load B
    tl.store(T_add_ptr + offsets, a + b, mask=mask)  # Store result


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mirrors the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (>= size,) and dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of shape (>= size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of shape (>= size,) and dtype torch.float32 on CUDA.
    size : int
        Number of elements to process.
    """
    # Basic sanity checks
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must have dtype torch.float32.")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise RuntimeError("Tensor sizes must be at least `size`.")

    BLOCK_SIZE = 320                # Threads per program (matches launch bounds)
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel.
    # `num_warps=10` gives 10 warps * 32 threads = 320 threads per program.
    _triton_kernel_impl[grid](
        A, B, C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=10,
    )