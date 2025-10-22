import torch
import triton
import triton.language as tl

# -------------------------------------------------------------
# Triton kernel: FP16 (A, B) → FP32 (C) GEMM using WMMA tiles
# -------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A, B, C,                     # pointers
    M, K, N,                     # matrix dimensions
    stride_am, stride_ak,        # A strides
    stride_bk, stride_bn,        # B strides
    stride_cm, stride_cn,        # C strides
    BLOCK_M: tl.constexpr,       # tile size in M dimension (rows)
    BLOCK_N: tl.constexpr,       # tile size in N dimension (cols)
    BLOCK_K: tl.constexpr,       # tile size in K dimension (inner)
):
    """
    Compute C = A @ B where:
        A: (M, K) half
        B: (K, N) half
        C: (M, N) float
    The kernel uses 16×16×16 WMMA tiles and accumulates in FP32.
    """
    pid_m = tl.program_id(1)  # block row index (y dimension)
    pid_n = tl.program_id(0)  # block column index (x dimension)

    # ---------------------------------------------------------
    # Offsets for the current tile
    # ---------------------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for out‑of‑bounds rows / columns
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---------------------------------------------------------
    # Loop over K dimension in steps of BLOCK_K
    # ---------------------------------------------------------
    for k_offset in range(0, K, BLOCK_K):
        offs_k = k_offset + tl.arange(0, BLOCK_K)

        # --------------------
        # Load A tile (M×K)
        # --------------------
        a_ptr = A + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        mask_a = mask_m[:, None] & (offs_k[None, :] < K)
        a = tl.load(a_ptr, mask=mask_a, other=0.0, dtype=tl.float16)

        # --------------------
        # Load B tile (K×N)
        # --------------------
        b_ptr = B + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)
        mask_b = (offs_k[:, None] < K) & mask_n[None, :]
        b = tl.load(b_ptr, mask=mask_b, other=0.0, dtype=tl.float16)

        # --------------------
        # Multiply‑accumulate using Tensor Cores
        # --------------------
        # tl.dot maps to WMMA when shapes are 16×16×16 and out_dtype is FP32
        acc += tl.dot(a, b, out_dtype=tl.float32)

    # ---------------------------------------------------------
    # Write the result back to C (FP32)
    # ---------------------------------------------------------
    c_ptr = C + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptr, acc, mask=mask_c)


# -------------------------------------------------------------
# Wrapper that matches the original CUDA kernel signature
# -------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                  m: int, k: int, n: int):
    """
    Triton wrapper mirroring the original CUDA kernel:
        triton_kernel(half *A, half *B, float *C, int m, int k, int n)

    Parameters
    ----------
    A : torch.Tensor
        Input matrix of shape (m, k) with dtype torch.float16.
    B : torch.Tensor
        Input matrix of shape (k, n) with dtype torch.float16.
    C : torch.Tensor
        Output matrix of shape (m, n) with dtype torch.float32.
    m, k, n : int
        Matrix dimensions.
    """
    # ---------------------------------------------------------
    # Basic sanity checks
    # ---------------------------------------------------------
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float16 and B.dtype == torch.float16, "A and B must be half precision"
    assert C.dtype == torch.float32, "C must be single precision"
    assert A.shape == (m, k) and B.shape == (k, n) and C.shape == (m, n), \
        "Tensor shapes must match the provided dimensions"

    # ---------------------------------------------------------
    # Row‑major strides (in elements)
    # ---------------------------------------------------------
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # ---------------------------------------------------------
    # Tile sizes (must match WMMA tile size)
    # ---------------------------------------------------------
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    # ---------------------------------------------------------
    # Grid configuration: (N‑tiles, M‑tiles)
    # ---------------------------------------------------------
    grid = (triton.cdiv(n, BLOCK_N), triton.cdiv(m, BLOCK_M))

    # ---------------------------------------------------------
    # Launch the kernel
    # ---------------------------------------------------------
    _triton_kernel_impl[grid](
        A, B, C,
        m, k, n,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,          # 4 warps (128 threads) per block
        # num_stages=3        # optional: enable pipelining
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()


# -------------------------------------------------------------
# Self‑test (optional)
# -------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    m, k, n = 1024, 128, 128
    A = torch.randn(m, k, dtype=torch.float16, device="cuda")
    B = torch.randn(k, n, dtype=torch.float16, device="cuda")
    C = torch.empty(m, n, dtype=torch.float32, device="cuda")

    # Reference result using PyTorch (FP32 accumulation)
    C_ref = torch.matmul(A.float(), B.float())

    # Run Triton implementation
    triton_kernel(A, B, C, m, k, n)

    # Verify correctness
    max_err = (C - C_ref).abs().max()
    print(f"Max absolute error vs PyTorch FP32 matmul: {max_err.item():.3e}")