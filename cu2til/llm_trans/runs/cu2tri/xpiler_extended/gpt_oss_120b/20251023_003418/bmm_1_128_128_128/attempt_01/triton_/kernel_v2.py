import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel implementing half-precision WMMA (FP16 x FP16 -> FP32)
# ---------------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr, B_ptr, C_ptr,
    M, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    K: tl.constexpr
):
    """
    Compute C = A @ B where
        A: (M, K) half
        B: (K, N) half
        C: (M, N) float
    Tile size is BLOCK_M x BLOCK_N with inner reduction BLOCK_K.
    """
    pid_m = tl.program_id(1)  # blockIdx.y
    pid_n = tl.program_id(0)  # blockIdx.x

    # Offsets for the output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in steps of BLOCK_K
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load A tile (BLOCK_M x BLOCK_K) in FP16
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0, dtype=tl.float16)

        # Load B tile (BLOCK_K x BLOCK_N) in FP16
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0, dtype=tl.float16)

        # WMMA-style matrix multiply: accumulate in FP32
        acc += tl.dot(a, b, out_dtype=tl.float32)

    # Store the result tile to C (FP32)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ---------------------------------------------------------------------------
# Wrapper function matching the original CUDA kernel signature
# ---------------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                  b: int, m: int, k: int, n: int):
    """
    Entry point mirroring the original CUDA kernel signature:
        triton_kernel(half *A, half *B, float *C, int b, int m, int k, int n)

    Parameters
    ----------
    A : torch.Tensor (torch.float16) of shape (m, k)
    B : torch.Tensor (torch.float16) of shape (k, n)
    C : torch.Tensor (torch.float32) of shape (m, n)
    b : int
        Batch dimension (currently unused, kept for API compatibility)
    m, k, n : int
        Matrix dimensions.
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float16 and B.dtype == torch.float16, "A and B must be half precision"
    assert C.dtype == torch.float32, "C must be single precision"
    assert A.shape == (m, k), f"A shape mismatch: expected ({m},{k}), got {A.shape}"
    assert B.shape == (k, n), f"B shape mismatch: expected ({k},{n}), got {B.shape}"
    assert C.shape == (m, n), f"C shape mismatch: expected ({m},{n}), got {C.shape}"

    # Strides in elements (row-major)
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Tile sizes – must match the constexprs in the kernel
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    # Grid configuration (program IDs)
    grid = ((n + BLOCK_N - 1) // BLOCK_N, (m + BLOCK_M - 1) // BLOCK_M)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        m, n,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        K=k,
        num_warps=4  # 4 warps = 128 threads per block (fits a 16x16 tile)
    )
    # Ensure completion before returning to the caller
    torch.cuda.synchronize()