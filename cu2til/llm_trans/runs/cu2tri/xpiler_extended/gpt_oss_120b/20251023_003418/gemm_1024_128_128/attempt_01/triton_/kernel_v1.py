import torch
import triton
import triton.language as tl

# Tile sizes (must match the WMMA tile size 16×16×16)
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 16

@triton.jit
def _triton_kernel_impl(
    A, B, C,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn
):
    """
    Matrix multiplication using Tensor Cores (FP16 → FP32).

    Computes C = A @ B where
      A: (M, K) half
      B: (K, N) half
      C: (M, N) float
    """
    pid_m = tl.program_id(1)  # block row index
    pid_n = tl.program_id(0)  # block column index

    # Global row/col indices for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for out‑of‑bounds rows/cols
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension in steps of BLOCK_K
    for k_offset in range(0, K, BLOCK_K):
        offs_k = k_offset + tl.arange(0, BLOCK_K)

        # Load A tile (M×K) as FP16
        a_ptr = A + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        mask_a = mask_m[:, None] & (offs_k[None, :] < K)
        a = tl.load(a_ptr, mask=mask_a, other=0.0, dtype=tl.float16)

        # Load B tile (K×N) as FP16
        b_ptr = B + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)
        mask_b = (offs_k[:, None] < K) & mask_n[None, :]
        b = tl.load(b_ptr, mask=mask_b, other=0.0, dtype=tl.float16)

        # FP16×FP16 → FP32 dot product
        acc += tl.dot(a, b, out_dtype=tl.float32)

    # Write the result back to C (FP32)
    c_ptr = C + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptr, acc, mask=mask_c)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                  m: int, k: int, n: int):
    """
    Triton wrapper that mirrors the original CUDA kernel signature.

    Parameters
    ----------
    A : torch.Tensor
        Input matrix of shape (m, k) with dtype torch.float16.
    B : torch.Tensor
        Input matrix of shape (k, n) with dtype torch.float16.
    C : torch.Tensor
        Output matrix of shape (m, n) with dtype torch.float32.
    m, k, n : int
        Dimensions of the matrices.
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float16 and B.dtype == torch.float16, "A and B must be half precision"
    assert C.dtype == torch.float32, "C must be single precision"
    assert A.shape == (m, k) and B.shape == (k, n) and C.shape == (m, n), "Tensor shapes must match the provided dimensions"

    # Row‑major strides (in elements)
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Grid configuration: (N‑tiles, M‑tiles)
    grid = (triton.cdiv(n, BLOCK_N), triton.cdiv(m, BLOCK_M))

    # Launch the kernel (one warp per block)
    _triton_kernel_impl[grid](
        A, B, C,
        m, k, n,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        num_warps=1
    )
    # Ensure the kernel has finished before returning
    torch.cuda.synchronize()