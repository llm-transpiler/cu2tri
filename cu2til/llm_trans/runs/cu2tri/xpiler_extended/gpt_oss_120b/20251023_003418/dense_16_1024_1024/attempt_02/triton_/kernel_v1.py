import torch
import triton
import triton.language as tl

# Tuned block sizes for the H800 (Hopper) GPU
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
    num_warps: tl.constexpr
):
    pid_m = tl.program_id(0)  # block row index
    pid_n = tl.program_id(1)  # block column index

    # Absolute indices for this block
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for out‑of‑bounds rows/cols
    row_mask = row < M
    col_mask = col < N

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles of size BLOCK_K
    k = 0
    while k < K:
        k_idx = k + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K

        # Load A tile (M×K)
        a_ptrs = a_ptr + (row[:, None] * stride_am) + (k_idx[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        a = a.to(tl.float32)

        # Load B tile (K×N)
        b_ptrs = b_ptr + (k_idx[:, None] * stride_bk) + (col[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        b = b.to(tl.float32)

        # Multiply‑accumulate
        acc += tl.dot(a, b)

        k += BLOCK_K

    # Add bias (broadcast over rows)
    bias = tl.load(bias_ptr + col, mask=col_mask, other=0.0)
    bias = bias[None, :]  # shape (1, BLOCK_N)
    acc = acc + bias

    # Write result back to global memory
    out_ptrs = out_ptr + (row[:, None] * stride_out_m) + (col[None, :] * stride_out_n)
    tl.store(out_ptrs, acc, mask=row_mask[:, None] & col_mask[None, :])

def triton_kernel(a: torch.Tensor,
                  b: torch.Tensor,
                  bias: torch.Tensor,
                  output: torch.Tensor,
                  M: int,
                  K: int,
                  N: int):
    """
    Triton implementation of the dense kernel:
        output = a @ b + bias

    Parameters
    ----------
    a : torch.Tensor
        Input matrix A of shape (M, K) with dtype torch.float16.
    b : torch.Tensor
        Input matrix B of shape (K, N) with dtype torch.float16.
    bias : torch.Tensor
        Bias vector of shape (N,) with dtype torch.float32.
    output : torch.Tensor
        Output matrix of shape (M, N) with dtype torch.float32.
    M, K, N : int
        Dimensions of the multiplication.
    """
    # Validate inputs
    assert a.is_cuda and b.is_cuda and bias.is_cuda and output.is_cuda, "All tensors must be on CUDA"
    assert a.dtype == torch.float16 and b.dtype == torch.float16, "a and b must be FP16"
    assert bias.dtype == torch.float32 and output.dtype == torch.float32, "bias and output must be FP32"
    assert a.shape == (M, K), f"Expected a.shape ({M},{K}), got {a.shape}"
    assert b.shape == (K, N), f"Expected b.shape ({K},{N}), got {b.shape}"
    assert bias.shape == (N,), f"Expected bias.shape ({N},), got {bias.shape}"
    assert output.shape == (M, N), f"Expected output.shape ({M},{N}), got {output.shape}"

    # Ensure contiguous layout for simple stride handling
    if not a.is_contiguous():
        a = aiguous()
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
    stream = torch.cuda.current_stream().cuda_stream
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
       =BLOCK_K,
        num_warps=8,
        stream=stream,
    )