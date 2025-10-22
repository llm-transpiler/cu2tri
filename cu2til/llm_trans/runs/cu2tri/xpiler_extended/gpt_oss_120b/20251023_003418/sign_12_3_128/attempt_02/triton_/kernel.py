import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, C_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes the sign of each element in A and writes the result to C.
    The sign is defined as:
        1.0  if A[i] > 0
       -1.0  if A[i] < 0
        0.0  if A[i] == 0

    Parameters
    ----------
    A_ptr : pointer to float32
        Input array.
    C_ptr : pointer to float32
        Output array where the sign of each element is stored.
    size : int
        Number of elements to process.
    BLOCK_SIZE : int (constexpr)
        Number of threads per block (workgroup).
    """
    pid = tl.program_id(0)  # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices
    mask = offsets < size  # guard against out‑of‑bounds

    # Load input values (masked loads return 0.0 for out‑of‑bounds elements)
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # Compute sign using nested tl.where:
    #   1.0 if a > 0
    #  -1.0 if a < 0
    #   0.0 otherwise
    sign = tl.where(a > 0.0, 1.0, tl.where(a < 0.0, -1.0, 0.0))

    # Store the result (masked store)
    tl.store(C_ptr + offsets, sign, mask=mask)

def triton_kernel(A: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that launches the Triton kernel.
    Mirrors the original CUDA kernel signature.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of the same shape and dtype as A.
    size : int
        Logical number of elements to process.
    """
    # Validation
    assert A.is_cuda and C.is_cuda, "A and C must be CUDA tensors"
    assert A.dtype == torch.float32 and C.dtype == torch.float32, "Tensors must be float32"
    assert A.numel() >= size and C.numel() >= size, "Tensor storage must be at least 'size' elements"

    if size == 0:
        return

    BLOCK_SIZE = 1024
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch kernel
    _triton_kernel_impl[grid](A, C, size, BLOCK_SIZE=BLOCK_SIZE)

# Quick sanity check
if __name__ == "__main__":
    size = 4608
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    triton_kernel(A, C, size)

    expected = torch.sign(A)
    if torch.allclose(C, expected):
        print("Triton kernel produced correct results.")
    else:
        print("Discrepancy detected:", (C - expected).abs().max().item())