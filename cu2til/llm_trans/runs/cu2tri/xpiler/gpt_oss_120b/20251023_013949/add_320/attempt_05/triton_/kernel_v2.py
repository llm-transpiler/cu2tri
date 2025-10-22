import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise addition.
# Must be named exactly as required.
@triton.jit
def _triton_kernel_impl(
    A_ptr,          # float* __restrict__ A
    B_ptr,          # float* __restrict__ B
    C_ptr,          # float* __restrict__ T_add (output)
    N,              # total number of elements to process
    BLOCK_SIZE: tl.constexpr  # compile‑time constant (power of 2)
):
    pid = tl.program_id(0)                     # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global offsets
    mask = offsets < N                         # guard out‑of‑range threads

    a = tl.load(A_ptr + offsets, mask=mask)    # load A
    b = tl.load(B_ptr + offsets, mask=mask)    # load B
    tl.store(C_ptr + offsets, a + b, mask=mask)  # store result


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point mirroring the original CUDA kernel signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA).
    B : torch.Tensor
        Input tensor B (float32, CUDA).
    C : torch.Tensor
        Output tensor C (float32, CUDA). Must have at least `size` elements.
    size : int
        Number of elements to process.
    """
    # Basic sanity checks – keep behavior identical to the CUDA version.
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on the CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    assert A.numel() >= size and B.numel() >= size and C.numel() >= size, \
        "Tensor storage must be at least `size` elements"
    if size == 0:
        return

    # Triton prefers block sizes that are powers of two.
    BLOCK_SIZE = 256  # power‑of‑2 close to the original 320

    # Compute grid size exactly as in the CUDA host code (but using the new BLOCK_SIZE).
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel.
    _triton_kernel_impl[grid](A, B, C, size, BLOCK_SIZE=BLOCK_SIZE)

    # Optional: synchronize to make kernel completion explicit (not required for correctness).
    torch.cuda.synchronize()