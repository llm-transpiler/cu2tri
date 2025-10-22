import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, x_ptr, y_ptr, m, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes y[row] = sum_i A[row, i] * x[i] for row < m.
    Parameters:
        A_ptr: pointer to matrix A (shape [m, n])
        x_ptr: pointer to vector x (shape [n])
        y_ptr: pointer to output vector y (shape [m])
        m: number of rows (runtime)
        n: number of columns (compile‑time constant)
        BLOCK_SIZE: tile size for the column dimension (compile‑time constant)
    """
    row = tl.program_id(0)

    # Guard against out‑of‑bounds rows
    if row >= m:
        return

    # Accumulator for the dot product (scalar)
    acc = tl.zeros([], dtype=tl.float32)

    # Loop over the column dimension in BLOCK_SIZE‑wide tiles
    for col in range(0, n, BLOCK_SIZE):
        # Compute base pointers for this tile
        a_tile_ptr = A_ptr + row * n + col
        x_tile_ptr = x_ptr + col

        # Offsets within the tile
        offset = tl.arange(0, BLOCK_SIZE)

        # Mask for the tail of the last tile
        col_mask = (col + offset) < n

        # Load a tile of A and the corresponding slice of x
        a_tile = tl.load(a_tile_ptr + offset, mask=col_mask, other=0.0)
        x_tile = tl.load(x_tile_ptr + offset, mask=col_mask, other=0.0)

        # Accumulate dot product for this tile
        acc += tl.dot(a_tile, x_tile)

    # Write the result back to y
    tl.store(y_ptr + row, acc)


def triton_kernel(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, m: int, n: int):
    """
    Entry‑point wrapper that launches the Triton kernel.
    Arguments must match the original CUDA kernel signature.
    """
    # Basic validation
    if not (A.is_cuda and x.is_cuda and y.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if A.dtype != torch.float32 or x.dtype != torch.float32 or y.dtype != torch.float32:
        raise RuntimeError("All tensors must be torch.float32")
    # Ensure contiguous layout
    A = A.contiguous()
    x = x.contiguous()
    y = y.contiguous()

    # Tile size for the column dimension (matches the original kernel)
    BLOCK_SIZE = 128

    # One program (thread) per output row
    grid = (m,)

    # Launch the kernel. `n` is a compile‑time constant, passed as a keyword argument.
    _triton_kernel_impl[grid](
        A,
        x,
        y,
        m,
        n=n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,   # good occupancy on H800
    )


# Self‑test
if __name__ == "__main__":
    torch.manual_seed(0)
    m, n = 112, 128
    A = torch.randn(m, n, device="cuda", dtype=torch.float32)
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.zeros(m, device="cuda", dtype=torch.float32)

    # Run Triton implementation
    triton_kernel(A, x, y, m, n)

    # Reference result using PyTorch
    y_ref = torch.matmul(A, x)

    # Verify correctness
    max_err = (y - y_ref).abs().max()
    print(f"Max error: {max_err.item():.3e}")