import torch
import triton
import triton.language as tl

# Tile sizes (must be multiples of 16 for Tensor Core usage)
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 16

@triton.jit
def _triton_kernel_impl(
    A, B, C,
    M, N, K,
    stride_a_batch, stride_am, stride_ak,
    stride_b_batch, stride_bk, stride_bn,
    stride_c_batch, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_n = tl.program_id(0)  # column block index
    pid_m = tl.program_id(1)  # row block index
    pid_b = tl.program_id(2)  # batch index

    # Offsets for the output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_offset in range(0, K, BLOCK_K):
        offs_k = tl.arange(0, BLOCK_K)

        # Pointers to A and B tiles (including batch offset)
        a_ptr = A + pid_b * stride_a_batch + (offs_m[:, None] * stride_am + (k_offset + offs_k)[None, :] * stride_ak)
        b_ptr = B + pid_b * stride_b_batch + ((k_offset + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Load tiles with boundary checks
        a = tl.load(
            a_ptr,
            mask=(offs_m[:, None] < M) & ((k_offset + offs_k)[None, :] < K),
            other=0.0,
            dtype=tl.float16,
        )
        b = tl.load(
            b_ptr,
            mask=((k_offset + offs_k)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
            dtype=tl.float16,
        )

        # Compute and accumulate
        acc = tl.dot(a, b, c=acc, out_dtype=tl.float32)

    # Store the result
    c_ptr = C + pid_b * stride_c_batch + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr, acc, mask=mask)


def triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    b: int,
    m: int,
    k: int,
    n: int,
):
    """
    Triton entry point mirroring the original CUDA kernel signature.
    Parameters:
        A (torch.Tensor): half‑precision matrix (m×k) or batched (b,m,k).
        B (torch.Tensor): half‑precision matrix (k×n) or batched (b,k,n).
        C (torch.Tensor): single‑precision output matrix (m×n) or batched (b,m,n).
        b (int): batch size.
        m (int): number of rows of A and C.
        k (int): inner dimension.
        n (int): number of columns of B and C.
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert (
        A.dtype == torch.float16
        and B.dtype == torch.float16
        and C.dtype == torch.float32
    ), "Expected dtypes: A,B half; C float32"

    # Determine whether tensors are batched
    if A.dim() == 3:
        assert A.shape == (b, m, k), f"A shape {A.shape} != ({b},{m},{k})"
        stride_a_batch = A.stride(0)
        stride_am = A.stride(1)
        stride_ak = A.stride(2)
    else:
        assert A.shape == (m, k), f"A shape {A.shape} != ({m},{k})"
        stride_a_batch = 0
        stride_am = A.stride(0)
        stride_ak = A.stride(1)

    if B.dim() == 3:
        assert B.shape == (b, k, n), f"B shape {B.shape} != ({b},{k},{n})"
        stride_b_batch = B.stride(0)
        stride_bk = B.stride(1)
        stride_bn = B.stride(2)
    else:
        assert B.shape == (k, n), f"B shape {B.shape} != ({k},{n})"
        stride_b_batch = 0
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)

    if C.dim() == 3:
        assert C.shape == (b, m, n), f"C shape {C.shape} != ({b},{m},{n})"
        stride_c_batch = C.stride(0)
        stride_cm = C.stride(1)
        stride_cn = C.stride(2)
    else:
        assert C.shape == (m, n), f"C shape {C.shape} != ({m},{n})"
        stride_c_batch = 0
        stride_cm = C.stride0)
        stride_cn = C.stride(1)

    # Grid dimensions (ceil division)
    grid_x = triton.cdiv(n, BLOCK_N)
    grid_y = triton.cdiv(m, BLOCK_M)
    grid_z = b  # batch dimension

    grid = (grid_x, grid_y, grid_z)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A,
        B,
        C,
        m,
        n,
        k,
        stride_a_batch,
        stride_am,
        stride_ak,
        stride_b_batch,
        stride_bk,
        stride_bn,
        stride_c_batch,
        stride_cm,
        stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    torch.cuda.synchronize()


# Self‑test (run this block to verify correctness)
if __name__ == "__main__":
    torch.manual_seed(0)

    # Test parameters
    b = 1
    m = 128
    k = 128
    n = 128

    # Batched tensors (shape (b, m, k), (b, k, n), (b, m, n A = torch.randn(b, m, k, dtype=torch.float16, device="cuda")
    B = torch.randn(b, k, n, dtype=torch.float16, device="cuda")
    C = torch.empty(b, m, n, dtype=torch.float32, device="cuda")

    triton_kernel(A, B, C, b, m, k, n)

    C_ref = torch.bmm(A.to(torch.float32), B.to(torch.float32))
    max_err = (C - C_ref).abs().max().item()
    print(f"Max absolute error (batched): {max_err:e}")

    # Non‑batched tensors (shape (m, k), (k, n), (m, n))
    A2 = torch.randn(m, k, dtype=torch.float16, device="cuda")
    B2 = torch.randn(k, n, dtype=torch.float16, device="cuda")
    C2 = torch.empty(m, n, dtype=torch.float32, device="cuda")

    triton_kernel(A2, B2, C2, b=1, m=m, k=k, n=n)

    C2_ref = torch.matmul(A2.to(torch.float32), B2.to(torch.float32))
    max_err2 = (C2 - C2_ref).abs().max().item()
    print(f"Max absolute error (non‑batched): {max_err2:e}")