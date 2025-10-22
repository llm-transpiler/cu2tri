import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    A_ptr, x_ptr, y_ptr,
    m, n,
    BLOCK: tl.constexpr,
    NUM_ITER: tl.constexpr
):
    """Compute y = A @ x for a single row (program ID)."""
    row = tl.program_id(0)
    if row >= m:
        return

    acc = tl.float32(0)
    for i in range(NUM_ITER):
        offset = i * BLOCK
        cols = tl.arange(0, BLOCK) + offset
        mask = cols < n

        a = tl.load(A_ptr + row * n + cols, mask=mask, other=0.0)
        x_vec = tl.load(x_ptr + cols, mask=mask, other=0.0)

        acc += tl.sum(a * x_vec)

    tl.store(y_ptr + row, acc)

def triton_kernel(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, m: int, n: int):
    """
    Entry point matching the original CUDA kernel signature.
    Computes y = A @ x where A is (m, n), x is (n,), y is (m,).
    """
    # Validate inputs
    assert A.is_cuda and x.is_cuda and y.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and x.dtype == torch.float32 and y.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.shape == (m, n), f"A shape mismatch: expected ({m}, {n}), got {A.shape}"
    assert x.shape == (n,), f"x shape mismatch: expected ({n},), got {x.shape}"
    assert y.shape == (m,), f"y shape mismatch: expected ({m},), got {y.shape}"

    # Ensure contiguous memory for optimal loads
    A = A.contiguous()
    x = x.contiguous()
    y = y.contiguous()

    # Inner-loop block size (matches original inner loop length)
    BLOCK = 128
    NUM_ITER = (n + BLOCK - 1) // BLOCK

    # One program per output row
    grid = (m,)

    _triton_kernel_impl[grid](
        A, x, y,
        m, n,
        BLOCK=BLOCK,
        NUM_ITER=NUM_ITER,
        num_warps=4  # Good default for FP32 on H800
    )