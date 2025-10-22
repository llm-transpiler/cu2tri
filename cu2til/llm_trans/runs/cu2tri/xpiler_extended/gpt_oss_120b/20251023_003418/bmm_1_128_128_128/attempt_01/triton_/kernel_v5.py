import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel implementing half‑precision matrix multiply (FP16×FP16→FP32)
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
    Tile size is BLOCK_M × BLOCK_N with inner reduction BLOCK_K.
    """
    pid_m = tl.program_id(1)  # blockIdx.y
    pid_n = tl.program_id(0)  # blockIdx.x

    # Offsets for the output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in steps of BLOCK_K
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load A tile (BLOCK_M × BLOCK_K) in FP16
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # dtype inferred from pointer (half)

        # Load B tile (BLOCK_K × BLOCK_N) in FP16
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # dtype inferred from pointer (half)

        # Accumulate in FP32
        acc += tl.dot(a, b, out_dtype=tl.float32)

    # Store the result tile to C (FP32)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ---------------------------------------------------------------------------
# Wrapper matching the original CUDA kernel signature
# ---------------------------------------------------------------------------
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                  b: int, m: int, k: int, n: int):
    """
    Entry point mirroring the original CUDA kernel signature:
        triton_kernel(half *A, half *B, float *C, int b, int m, int k, int n)

    Parameters
    ----------
    A : torch.Tensor (torch.float16) of shape (b, m, k) or (m, k)
    B : torch.Tensor (torch.float16) of shape (b, k, n) or (k, n)
    C : torch.Tensor (torch.float32) of shape (b, m, n) or (m, n)
    b, m, k, n : matrix dimensions (batch, rows, inner, cols)
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float16 and B.dtype == torch.float16, "A and B must be half precision"
    assert C.dtype == torch.float32, "C must be single precision"

    # Determine whether inputs include a batch dimension
    batched = (A.dim() == 3) or (B.dim() == 3) or (C.dim() == 3)

    if batched:
        # Expect explicit batch dimension matching the provided `b`
        assert A.shape == (b, m, k), f"A shape mismatch: expected ({b},{m},{k}), got {A.shape}"
        assert B.shape == (b, k, n), f"B shape mismatch: expected ({b},{k},{n}), got {B.shape}"
        assert C.shape == (b, m, n), f"C shape mismatch: expected ({b},{m},{n}), got {C.shape}"
    else:
        # No batch dimension; treat as a single batch
        assert A.shape == (m, k), f"A shape mismatch: expected ({m},{k}), got {A.shape}"
        assert B.shape == (k, n), f"B shape mismatch: expected ({k},{n}), got {B.shape}"
        assert C.shape == (m, n), f"C shape mismatch: expected ({m},{n}), got {C.shape}"
        # Add a dummy batch dimension for uniform handling
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
        C = C.unsqueeze(0)
        b = 1

    # Tile sizes – must match the WMMA tile of 16×16×16
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    # Grid configuration (one program per output tile)
    grid = ((n + BLOCK_N - 1) // BLOCK_N, (m + BLOCK_M - 1) // BLOCK_M)

    # Process each batch sequentially
    for batch_idx in range(b):
        A_i = A[batch_idx]
        B_i = B[batch_idx]
        C_i = C[batch_idx]

        # Strides in elements (row‑major)
        stride_am, stride_ak = A_i.stride()
        stride_bk, stride_bn = B_i.stride()
        stride_cm, stride_cn = C_i.stride()

        # Launch the Triton kernel for this batch
        _triton_kernel_impl[grid](
            A_i, B_i, C_i,
            m, n,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            K=k,
            num_warps=4  # 4 warps = 128 threads per block (fits a 16×16 tile)
        )

    # Ensure all kernels have finished before returning
    torch.cuda.synchronize()