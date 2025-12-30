import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=1),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def persistent_matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride variables
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """简化的Triton persistent matrix multiplication kernel"""
    # Program ID maps to a specific block in the output matrix
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Skip if out of bounds
    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return

    # Compute block starting positions
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Masks for boundary conditions
    mask_m = offs_am < M
    mask_n = offs_bn < N

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Persistent loop over K dimension
    for start_k in tl.range(0, K, BLOCK_SIZE_K):
        # Load A block
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + (start_k + offs_k[None, :]) * stride_ak
        mask_k = (start_k + offs_k) < K
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B block
        b_ptrs = b_ptr + (start_k + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Matrix multiplication
        accumulator = tl.dot(a, b, accumulator)

    # Store result
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    c = accumulator.to(tl.float16)
    tl.store(c_ptrs, c, mask=c_mask)

def triton_kernel(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                  M: int, N: int, K: int,
                  stride_am: int, stride_ak: int,
                  stride_bk: int, stride_bn: int,
                  stride_cm: int, stride_cn: int) -> torch.Tensor:
    """Triton实现：persistent matrix multiplication"""
    # Ensure inputs are contiguous
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()

    # Launch kernel
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    persistent_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn
    )

    return c