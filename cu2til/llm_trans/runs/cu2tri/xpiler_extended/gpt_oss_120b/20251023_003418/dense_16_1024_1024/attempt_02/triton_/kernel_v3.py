import torch
import triton
import triton.language as tl

# Tunable block sizes for the H800 (Hopper) GPU
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32

@triton.jit
def _triton_kernel_impl(
    a_ptr, b_ptr, bias_ptr, out_ptr,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_bias,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
):
    pid_m = tl.program_id(0)  # block row index
    pid_n = tl.program_id(1)  # block column index

    # Absolute row/col indices for this block
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for out-of-bounds handling
    row_mask = rows < M
    col_mask = cols < N

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles of size BLOCK_K
    k = 0
    while k < K:
        ks = k + tl.arange(0, BLOCK_K)
        k_mask = ks < K

        # Load A tile (M x K)
        a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
        a = tl.load(
            a_ptrs,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Load B tile (K x N)
        b_ptrs = b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Multiply‑accumulate
        acc += tl.dot(a, b)

        k += BLOCK_K

    # Add bias (broadcast over rows)
    bias = tl.load(bias_ptr + cols * stride_bias, mask=col_mask, other=0.0)
    bias = bias[None, :]  # shape (1, BLOCK_N)
    acc = acc + bias

    # Write result back to global memory
    out_ptrs = out_ptr + rows[:, None] * stride_out_m + cols[None, :] * stride_out_n
    tl.store(
        out_ptrs,
        acc,
        mask=row_mask[:, None] & col_mask[None, :],
    )

def triton_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    M: int,
    K: int,
    N: int,
):
    """
    Triton implementation of the dense kernel:
        output = a @ b + bias

    Parameters
    ----------
    a : torch.Tensor of shape (M, K), dtype torch.float16
    b : torch.Tensor of shape (K, N), dtype torch.float16
    bias : torch.Tensor of shape (N,), dtype torch.float32
    output : torch.Tensor of shape (M, N), dtype torch.float32
    M, K, N : int
        Dimensions of the matrix multiplication.
    """
    # Input validation
    assert a.is_cuda and b.is_cuda and bias.is_cuda and output.is_cuda, "All tensors must be CUDA tensors"
    assert a.dtype == torch.float16 and b.dtype == torch.float16, "a and b must be FP16"
    assert bias.dtype == torch.float32 and output.dtype == torch.float32, "bias and output must be FP32"
    assert a.shape == (M, K), f"a shape mismatch: expected ({M},{K}), got {a.shape}"
    assert b.shape == (K, N), f"b shape mismatch: expected ({K},{N}), got {b.shape}"
    assert bias.shape == (N,), f"bias shape mismatch: expected ({N},), got {bias.shape}"
    assert output.shape == (M, N), f"output shape mismatch: expected ({M},{N}), got {output.shape}"

    # Ensure contiguous layout
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()
    if not output.is_contiguous():
        output = output.contiguous()

    # Extract strides (in elements)
    stride_am = a.stride(0)
    stride_ak = a.stride(1)
    stride_bk = b.stride(0)
    stride_bn = b.stride(1)
    stride_bias = bias.stride(0)
    stride_out_m = output.stride(0)
    stride_out_n = output.stride(1)

    # Grid dimensions
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)

    # Launch kernel
    _triton_kernel_impl[(grid_m, grid_n)](
        a,
        b,
        bias,
        output,
        M,
        K,
        N,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_bias,
        stride_out_m,
        stride_out_n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
    )
    # The output tensor is modified in-place; no explicit return needed.