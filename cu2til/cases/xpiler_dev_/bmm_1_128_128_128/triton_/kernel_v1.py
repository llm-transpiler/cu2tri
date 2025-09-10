import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Basic configurations for a modern GPU like Ada
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_stages': 3, 'num_warps': 8}),
        # A configuration that resembles the CUDA kernel's 16x16 tiling
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16, 'num_stages': 2, 'num_warps': 1}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _triton_kernel_impl(
    A_ptr, B_ptr, C_ptr,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication C = A x B.
    This kernel is designed to be a high-performance replacement for the provided
    CUDA WMMA kernel. It computes a block of the output matrix C for each
    program instance.

    Args:
        A_ptr: Pointer to the input matrix A (M x K).
        B_ptr: Pointer to the input matrix B (K x N).
        C_ptr: Pointer to the output matrix C (M x N).
        M, K, N: Dimensions of the matrices.
        stride_am, stride_ak: Strides of matrix A.
        stride_bk, stride_bn: Strides of matrix B.
        stride_cm, stride_cn: Strides of matrix C.
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Compile-time constants for tiling.
    """
    # -----------------------------------------------------------
    # Map program ids to M and N blocks. This kernel uses a 2D grid.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance these pointers as we move in the K direction.
    # `offs_m`, `offs_n`, `offs_k` are ranges for the block dimensions.
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # `a_ptrs` is a block of pointers to the A matrix tile.
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # `b_ptrs` is a block of pointers to the B matrix tile.
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher precision, matching the CUDA kernel's behavior.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, handling boundary cases.
        # If K is not a multiple of BLOCK_SIZE_K, the last block will be partial.
        # We use masks to avoid out-of-bounds memory access.
        k_remaining = K - k * BLOCK_SIZE_K
        
        # Create masks for loading A and B tiles
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        
        # Load A and B tiles with padding (other=0.0) for masked-out elements
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Perform the matrix multiplication on the tiles and accumulate the result.
        accumulator = tl.dot(a, b, accumulator)
        
        # Advance the pointers to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # Write back the block of the output matrix C.
    # Create pointers to the C matrix tile.
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    # Create a mask to handle cases where M and N are not multiples of block sizes.
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, b: int, m: int, k: int, n: int):
    """
    Wrapper function for the Triton GEMM kernel, with a signature identical
    to the provided CUDA wrapper.

    Args:
        A (torch.Tensor): Input tensor A of shape (m, k) and dtype float16.
        B (torch.Tensor): Input tensor B of shape (k, n) and dtype float16.
        C (torch.Tensor): Output tensor C of shape (m, n) and dtype float32, modified in-place.
        b (int): Batch size (ignored, consistent with the provided CUDA kernel).
        m (int): Number of rows in A and C.
        k (int): Number of columns in A and rows in B.
        n (int): Number of columns in B and C.
    """
    # --- Assertions for input validation ---
    # Shape checks
    assert A.shape == (m, k), f"A shape mismatch: expected ({m}, {k}), got {A.shape}"
    assert B.shape == (k, n), f"B shape mismatch: expected ({k}, {n}), got {B.shape}"
    assert C.shape == (m, n), f"C shape mismatch: expected ({m}, {n}), got {C.shape}"
    # Dtype checks
    assert A.dtype == torch.float16, f"A dtype mismatch: expected float16, got {A.dtype}"
    assert B.dtype == torch.float16, f"B dtype mismatch: expected float16, got {B.dtype}"
    assert C.dtype == torch.float32, f"C dtype mismatch: expected float32, got {C.dtype}"
    # Device checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on a CUDA device"

    # --- Grid setup ---
    # The grid is defined as a lambda function of the kernel's metadata.
    # This allows the autotuner to select the best BLOCK_SIZEs and the grid
    # to be configured accordingly at launch time.
    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_SIZE_M']),
        triton.cdiv(n, META['BLOCK_SIZE_N']),
    )

    # --- Kernel launch ---
    # The kernel is launched with the specified grid.
    # Strides are passed explicitly to handle both contiguous and non-contiguous
    # tensors correctly.
    _triton_kernel_impl[grid](
        A, B, C,
        m, k, n,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )