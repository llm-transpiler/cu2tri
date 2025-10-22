import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 16,
    BLOCK_K: tl.constexpr = 16,
):
    # Program IDs correspond to output tile coordinates (col, row)
    pid_n = tl.program_id(0)  # column block index
    pid_m = tl.program_id(1)  # row block index

    # Starting indices of the tile this program will compute
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Pointers to the current A and B tiles
        a_ptrs = A_ptr + (offs_m[:, None] * K + offs_k[None, :])
        b_ptrs = B_ptr + (offs_k[:, None] * N + offs_n[None, :])

        # Load with out‑of‑bounds masking
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        # Tensor‑core matrix multiply (half → float accumulation)
        acc += tl.dot(a, b)

    # Write the result back to C
    c_ptrs = C_ptr + (offs_m[:, None] * N + offs_n[None, :])
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                  m: int, k: int, n: int):
    """
    Triton implementation of the CUDA WMMA kernel.
    Parameters
    ----------
    A : torch.Tensor (m, k), dtype=torch.float16, CUDA tensor
        Left matrix.
    B : torch.Tensor (k, n), dtype=torch.float16, CUDA tensor
        Right matrix.
    C : torch.Tensor (m, n), dtype=torch.float32, CUDA tensor
        Output matrix (will be overwritten).
    m, k, n : int
        Logical dimensions of the multiplication.
    """
    # Basic sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must reside on CUDA"
    assert A.dtype == torch.float16 and B.dtype == torch.float16, "A and B must be half precision"
    assert C.dtype == torch.float32, "C must be single precision"
    assert A.shape == (m, k), f"Expected A shape ({m},{k}), got {A.shape}"
    assert B.shape == (k, n), f"Expected B shape ({k},{n}), got {B.shape}"
    assert C.shape == (m, n), f"Expected C shape ({m},{n}), got {C.shape}"

    # Grid: one program (block) per 16×16 tile of C
    grid = ((n + 16 - 1) // 16, (m + 16 - 1) // 16)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C,
        m, n, k,
        BLOCK_M=16, BLOCK_N=16, BLOCK_K=16,
    )
    # Ensure completion before returning to the caller
    torch.cuda.synchronize()