import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, T_ptr, size1, size2,
                        BLOCK_SIZE: tl.constexpr,
                        COLS: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)
    row = pid * BLOCK_SIZE + lane
    row_mask = row < size1

    col = tl.arange(0, COLS)
    offsets = row[:, None] * COLS + col[None, :]
    mask = row_mask[:, None] & (col < COLS)

    a_vals = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    max_val = tl.max(a_vals, axis=1)
    exp_vals = tl.exp(a_vals - max_val[:, None])
    denom = tl.sum(exp_vals, axis=1)
    softmax_vals = exp_vals / denom[:, None]

    tl.store(T_ptr + offsets, softmax_vals, mask=mask)

def triton_kernel(A: torch.Tensor, C: torch.Tensor, size1: int, size2: int):
    """
    Triton implementation of the softmax kernel.
    A: input tensor of shape (size1, size2), dtype torch.float32, on CUDA.
    C: output tensor of same shape, dtype torch.float32, on CUDA.
    size1: number of rows.
    size2: number of columns (e.g., 128).
    """
    if not (A.is_cuda and C.is_cuda):
        raise RuntimeError("Input tensors must be CUDA tensors")
    if A.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("Tensors must be of type torch.float32")
    if A.shape != (size1, size2) or C.shape != (size1, size2):
        raise RuntimeError("Tensor shapes must match (size1, size2)")

    BLOCK_SIZE = 36
    grid = ((size1 + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _triton_kernel_impl[grid](
        A,
        C,
        size1,
        size2,
        BLOCK_SIZE=BLOCK_SIZE,
        COLS=size2,
        num_warps=2
    )
    torch.cuda.synchronize()

if __name__ == "__main__":
    # Example usage and verification
    size1 = 1024
    size2 = 128
    A = torch.randn(size1, size2, device='cuda', dtype=torch.float32)
    C = torch.empty_like(A)
    triton_kernel(A, C, size1, size2)

    # Reference implementation using PyTorch
    max_vals = torch.max(A, dim=1, keepdim=True).values
    exp_vals = torch.exp - max_vals)
    softmax = exp_vals / exp_vals.sum(dim=1, keepdim=True)
    torch.testing.assert_allclose(C, softmax, atol=1e-6, rtol=1e-5)
    print("Triton kernel matches PyTorch reference.")