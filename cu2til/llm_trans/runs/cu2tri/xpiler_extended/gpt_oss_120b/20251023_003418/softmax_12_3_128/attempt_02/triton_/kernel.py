import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, T_ptr, size1, size2,
                        BLOCK_N: tl.constexpr):
    # Each program processes one row of the input matrix
    row = tl.program_id(0)

    # Guard against out-of-range rows (should never trigger with the chosen grid)
    if row >= size1:
        return

    # Column indices (compile‑time constant, must be a power of two)
    col = tl.arange(0, BLOCK_N)

    # Mask for valid columns (size2 may be <= BLOCK_N)
    col_mask = col < size2

    # Linear offset for each element in the row
    offset = row * size2 + col

    # Load the input values (masked load)
    a = tl.load(A_ptr + offset, mask=col_mask, other=0.0)

    # Compute the maximum value in the row
    max_val = tl.max(a) # scalar reduction

    # Compute exponentials shifted by the max for numerical stability
    exp_val = tl.exp(a - max_val)

    # Compute the denominator (sum of exponentials)
    denom = tl.sum(exp_val)  # scalar reduction

    # Softmax output
    softmax = exp_val / denom

    # Store the result (masked store)
    tl.store(T_ptr + offset, softmax, mask=col_mask)

def triton_kernel(A: torch.Tensor, C: torch.Tensor, size1: int, size2: int):
    """
    Triton implementation of the per‑row softmax kernel.
    Parameters:
        A (torch.Tensor): Input tensor of shape (size1, size2), dtype torch.float32, on CUDA.
        C (torch.Tensor): Output tensor of the same shape as A.
        size1 (int): Number of rows.
        size2 (int): Number of columns (e.g., 128).
    """
    if not (A.is_cuda and C.is_cuda):
        raise RuntimeError("Input tensors must be CUDA tensors")
    if A.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("Tensors must be torch.float32")
    if A.shape != (size1, size2) or C.shape != (size1, size2):
        raise RuntimeError(f"Tensor shapes must be ({size1}, {size2})")

    # Compile‑time block size for the column dimension (must be a power of two)
    BLOCK_N = 128  # matches the expected column size in the original CUDA kernel

    # One program (block) per row
    grid = (size1,)

    _triton_kernel_impl[grid](
        A,
        C,
        size1,
        size2,
        BLOCK_N=BLOCK_N,
        num_warps=2,   # 2 warps = 64 threads, sufficient for BLOCK_N=128
    )
    torch.cuda.synchronize()

if __name__ == "__main__":
    # Example usage and verification
    size1 = 1024
    size2 = 128
    A = torch.randn(size1, size2, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    triton_kernel(A, C, size1, size2)

    # Reference implementation using PyTorch
    max_vals = torch.max(A, dim=1, keepdim=True).values
    exp_vals = torch.exp(A - max_vals)
    softmax = exp_vals / exp_vals.sum(dim=1, keepdim=True)

    torch.testing.assert_allclose(C, softmax, atol=1e-6, rtol=1e-5)
    print("Triton kernel matches PyTorch reference.")