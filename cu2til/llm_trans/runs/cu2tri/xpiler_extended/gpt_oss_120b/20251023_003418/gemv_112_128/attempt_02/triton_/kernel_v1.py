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
    """
    Triton implementation of y = A @ x.
    Each program instance computes one output element y[row].
    """
    row = tl.program_id(0)
    if row >= m:
        return

    acc = 0.0
    # Loop over the column dimension in blocks of size BLOCK
    for i in range(NUM_ITER):
        offset = i * BLOCK
        cols = tl.arange(0, BLOCK) + offset
        mask = cols < n

        a = tl.load(A_ptr + row * n + cols, mask=mask, other=0.0)
        x_vec = tl.load(x_ptr + cols, mask=mask, other=0.0)

        acc += tl.dot(a, x_vec)

    tl.store(y_ptr + row, acc)


def triton_kernel(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, m: int, n: int):
    """
    Entry point matching the original CUDA kernel signature.
    Launches the Triton kernel to compute y = A @ x.
    """
    # Basic sanity checks
    assert A.is_cuda and x.is_cuda and y.is_cuda, "All tensors must reside on CUDA"
    assert A.dtype == torch.float32 and x.dtype == torch.float32 and y.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.shape == (m, n), f"A shape mismatch: expected ({m}, {n}), got {A.shape}"
    assert x.shape == (n,), f"x shape mismatch: expected ({n},), got {x.shape}"
    assert y.shape == (m,), f"y shape mismatch: expected ({m},), got {y.shape}"

    # Ensure contiguous memory layout for optimal loads
    A = A.contiguous()
    x = x.contiguous()
    y = y.contiguous()

    # Tunable block size for the column dimension (matches original inner loop)
    BLOCK = 128
    # Number of column blocks needed (compile‑time constant for the kernel)
    NUM_ITER = (n + BLOCK - 1) // BLOCK

    # One program per output row
    grid = (m,)

    # Launch the kernel
    _triton_kernel_impl[grid](
        A, x, y,
        m, n,
        BLOCK=BLOCK,
        NUM_ITER=NUM_ITER,
        num_warps=4  # good default for FP32 on H800
    )