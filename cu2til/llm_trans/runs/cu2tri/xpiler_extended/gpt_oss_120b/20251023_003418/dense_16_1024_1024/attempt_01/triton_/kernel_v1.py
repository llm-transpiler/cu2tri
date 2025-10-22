import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bias_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    """
    Triton kernel for dense matrix multiplication with bias.
    Computes: C = A @ B + bias
    A: (M, K) half
    B: (K, N) half
    bias: (N,) float32
    C: (M, N) float32
    """
    pid_m = tl.program_id(axis=1)  # block row index
    pid_n = tl.program_id(axis=0)  # block column index

    # Compute the start indices of the block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for out-of-bounds rows/cols
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Pointers to the current tiles of A and B
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

        # Load tiles with appropriate masking
        a = tl.load(a_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                    dtype=tl.float16)
        b = tl.load(b_ptrs,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                    dtype=tl.float16)

        # Cast to FP32 before multiplication to match __half2float behavior
        a_f32 = a.to(tl.float32)
        b_f32 = b.to(tl.float32)

        # Matrix multiply-accumulate
        acc += tl.dot(a_f32, b_f32)

    # Load bias (broadcast across rows)
    bias = tl.load(bias_ptr + offs_n * bias_stride,
                   mask=mask_n,
                   other=0.0,
                   dtype=tl.float32)
    acc += bias[None, :]  # broadcast bias to (BLOCK_M, BLOCK_N)

    # Write back the result
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptrs,
             acc,
             mask=mask_m[:, None] & mask_n[None, :])

def triton_kernel(a, b, bias, output, M, K, N):
    """
    Entry point that launches the Triton kernel.
    Parameters
    ----------
    a : torch.Tensor
        Input matrix A of shape (M, K) and dtype torch.float16.
    b : torch.Tensor
        Input matrix B of shape (K, N) and dtype torch.float16.
    bias : torch.Tensor
        Bias vector of shape (N,) and dtype torch.float32.
    output : torch.Tensor
        Output matrix C of shape (M, N) and dtype torch.float32.
    M, K, N : int
        Dimensions of the matrices.
    """
    # Basic sanity checks
    if not (a.is_cuda and b.is_cuda and bias.is_cuda and output.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise RuntimeError("A and B must be torch.float16")
    if bias.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("Bias and output must be torch.float32")

    # Ensure contiguous layout
    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()
    output = output.contiguous()

    # Strides in elements (not bytes)
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = output.stride()
    bias_stride = bias.stride(0)

    # Tunable block sizes (chosen for Hopper GPUs)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    num_warps = 8
    num_stages = 3

    # Grid dimensions
    grid = ((N + BLOCK_N - 1) // BLOCK_N,
            (M + BLOCK_M - 1) // BLOCK_M)

    # Launch kernel
    _triton_kernel_impl[grid](
        a,
        b,
        output,
        bias,
        M,
        K,
        N,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        bias_stride,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output

# ----------------------------------------------------------------------
# Simple correctness test (run only when this script is executed directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    M, K, N = 256, 512, 384
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    bias = torch.randn(N, dtype=torch.float32, device="cuda")
    out = torch.empty(M, N, dtype=torch.float32, device="cuda")

    # Run Triton implementation
    triton_kernel(a, b, bias, out, M, K, N)

    # Reference result using PyTorch (float32 computation)
    ref = a.float() @ b.float() + bias
    max_err = (out - ref).abs().max().item()
    print(f"Max absolute error vs PyTorch: {max_err:.3e}")