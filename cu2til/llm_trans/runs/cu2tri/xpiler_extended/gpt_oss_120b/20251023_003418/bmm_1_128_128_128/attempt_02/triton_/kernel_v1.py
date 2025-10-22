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
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(1)  # row block index
    pid_n = tl.program_id(0)  # column block index

    # Offsets for the output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles of size BLOCK_K
    for k_offset in range(0, K, BLOCK_K):
        offs_k = tl.arange(0, BLOCK_K)

        # Pointers to A and B tiles
        a_ptrs = A + (offs_m[:, None] * stride_am + (k_offset + offs_k)[None, :] * stride_ak)
        b_ptrs = B + ((k_offset + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Load tiles with boundary checks
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & ((k_offset + offs_k)[None, :] < K),
                    other=0.0,
                    dtype=tl.float16)
        b = tl.load(b_ptrs,
                    mask=((k_offset + offs_k)[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0,
                    dtype=tl.float16)

        # Matrix multiply and accumulate in FP32
        acc += tl.dot(a, b, out_dtype=tl.float32)

    # Write the result back to C with masking for out‑of‑bounds threads
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def triton_kernel(A: torch.Tensor,
                  B: torch.Tensor,
                  C: torch.Tensor,
                  b: int,
                  m: int,
                  k: int,
                  n: int):
    """
    Triton entry point mirroring the original CUDA kernel signature.
    Parameters:
        A (torch.Tensor): (m, k) half‑precision matrix.
        B (torch.Tensor): (k, n) half‑precision matrix.
        C (torch.Tensor): (m, n) single‑precision output matrix.
        b (int): batch size (currently unused, kept for API compatibility).
        m (int): number of rows of A and C.
        k (int): inner dimension.
        n (int): number of columns of B and C.
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA"
    assert A.dtype == torch.float16 and B.dtype == torch.float16 and C.dtype == torch.float32
    assert A.shape == (m, k)
    assert B.shape == (k, n)
    assert C.shape == (m, n)

    # Strides in elements (not bytes)
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Grid dimensions (ceil division)
    grid = (triton.cdiv(n, BLOCK_N), triton.cdiv(m, BLOCK_M))

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        m, n, k,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    # Ensure kernel completion before returning (optional but useful for timing)
    torch.cuda.synchronize()


# Self‑test (run this block to verify correctness)
if __name__ == "__main__":
    torch.manual_seed(0)
    m, k, n = 128, 128, 128  # dimensions matching the original CUDA example
    A = torch.randn(m, k, dtype=torch.float16, device="cuda")
    B = torch.randn(k, n, dtype=torch.float16, device="cuda")
    C = torch.empty(m, n, dtype=torch.float32, device="cuda")

    # Execute Triton kernel
    triton_kernel(A, B, C, b=1, m=m, k=k, n=n)

    # Reference result using PyTorch (FP16 inputs, FP32 accumulation)
    C_ref = torch.matmul(A.to(torch.float32), B.to(torch.float32))

    # Verify correctness
    max_err = (C - C_ref).abs().max().item()
    print(f"Max absolute error vs torch.matmul: {max_err:e}")