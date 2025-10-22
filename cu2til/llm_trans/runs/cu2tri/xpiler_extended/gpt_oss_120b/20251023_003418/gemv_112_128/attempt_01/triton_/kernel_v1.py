import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, x_ptr, y_ptr, m, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Program ID corresponds to the row index
    row = tl.program_id(0)
    # Mask for valid rows
    row_mask = row < m

    # Accumulator for the dot product (scalar)
    acc = tl.zeros([1], dtype=tl.float32)

    # Loop over the column dimension in chunks of BLOCK_SIZE
    for col in range(0, n, BLOCK_SIZE):
        # Pointers to the current block of A and x
        a_ptr = A_ptr + row * n + col
        x_ptr_offset = x_ptr + col

        # Offsets within the block
        offset = tl.arange(0, BLOCK_SIZE)

        # Mask for the tail of the last block
        col_mask = (col + offset) < n

        # Load a block of A and the corresponding part of x
        a = tl.load(a_ptr + offset, mask=col_mask, other=0.0)
        x = tl.load(x_ptr_offset + offset, mask=col_mask, other=0.0)

        # Accumulate the dot product of the two vectors
        acc += tl.dot(a, x, axis=0)

    # Write the result back to y if the row is valid
    tl.store(y_ptr + row, acc[0], mask=row_mask)

def triton_kernel(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, m: int, n: int):
    """
    Triton implementation of the CUDA kernel:
    y[row] = sum_{i=0}^{n-1} A[row, i] * x[i]   for row < m
    """
    # Basic validation
    if not (A.is_cuda and x.is_cuda and y.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if A.dtype != torch.float32 or x.dtype != torch.float32 or y.dtype != torch.float32:
        raise RuntimeError("All tensors must be torch.float32")
    # Ensure contiguous memory layout
    A = A.contiguous()
    x = x.contiguous()
    y = y.contiguous()

    # Choose a block size for the column dimension.
    # 128 works well on the H800 (80 GB SXM5) and matches the original kernel.
    BLOCK_SIZE = 128

    # One program per output row
    grid = (m,)

    # Launch the Triton kernel. `n` and `BLOCK_SIZE` are compile‑time constants.
    _triton_kernel_impl[grid](
        A,
        x,
        y,
        m,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,   # 4 warps per block gives good occupancy on the H800
    )

# Simple self‑test
if __name__ == "__main__":
    torch.manual_seed(0)
    m, n = 112, 128
    A = torch.randn(m, n, device="cuda", dtype=torch.float32)
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.zeros(m, device="cuda", dtype=torch.float32)

    # Run the Triton kernel
    triton_kernel(A, x, y, m, n)

    # Reference result using PyTorch
    y_ref = torch.matmul(A, x)

    # Verify correctness
    max_err = (y - y_ref).abs().max()
    print(f"Max error: {max_err.item():.3e}")