import torch
import triton
import triton.language as tl

# Triton kernel implementing FP16→FP32 GEMM using WMMA (16×16×16 tiles)
@triton.jit
def _triton_kernel_impl(
    A, B, C,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B where:
        A: (M, K) half
        B: (K, N) half
        C: (M, N) float
    The kernel uses 16×16×16 WMMA tiles and accumulates in FP32.
    """
    pid_m = tl.program_id(1)  # block row
    pid_n = tl.program_id(0)  # block column

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Out‑of‑bounds masks
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator (FP32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in steps of BLOCK_K
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

        # Multiply‑accumulate
        acc += tl.dot(a, b, out_dtype=tl.float32)

    # Store result (FP32)
    c_ptr = C + (offs_m[:, None] * stride_cm) + (offs_n[None, *_cn)
    mask_c mask_m[:, None] & mask[, :]
    tl.store_ptr, acc, mask=mask_c)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                  m: int, k: int, n: int):
    """
    Wrapper mirroring the original CUDA kernel signature:
        triton_kernel(half *A, half *B, float *C, int m, int k, int n)

    Parameters
    ----------
    A : torch.Tensor (m, k), dtype=torch.float16
    B : torch.Tensor (k, n), dtype=torch.float16
    C : torch.Tensor (m, n), dtype=torch.float32
    m, k, n : matrix dimensions
    """
    # Validation
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
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

    # Tile size (must match WMMA tile size)
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    # Grid: (N‑tiles, M‑tiles)
    grid = (triton.cdiv(n, BLOCK_N), triton.cdiv(m, BLOCK_M))

    # Launch kernel
    _triton_kernel_impl[grid](
        A, B, C,
        m, k, n,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4  # 4 warps (128 threads) per block
    )
    torch.cuda.synchronize()


# Optional self‑test (can be removed when integrating)
if __name__ == "__main__":
    torch.manual_seed(0)
    m, k, n = 1024, 128, 128
    A = torch.randn(m, k, dtype=torch.float16, device="cuda")
    B = torch.randn(k, n, dtype=torch.float16, device="cuda")
    C = torch.empty(m, n, dtype=torch.float32, device="cuda")
    # Reference using FP32 accumulation
    C_ref = torch.matmul(A.float(), B.float())
    # Run Triton kernel
    triton_kernel(A, B, C, m, k, n)
    # Verify
    max_err = (C - C_ref).abs().max()
    print(f"Max absolute error: {max_err.item()}")