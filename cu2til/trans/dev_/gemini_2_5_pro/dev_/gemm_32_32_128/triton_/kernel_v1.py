import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel_impl(
    # Pointers to matrices
    A, B, C,
    # Matrix dimensions
    M, K, N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication C = A * B.
    This kernel is a direct translation of the provided CUDA WMMA kernel.
    It computes a BLOCK_SIZE_M x BLOCK_SIZE_N tile of C per program instance.
    
    A is (M, K) of type float16
    B is (K, N) of type float16
    C is (M, N) of type float32
    """
    # -----------------------------------------------------------
    # Map program ids to M and N dimensions, which corresponds to CUDA's
    # blockIdx.y and blockIdx.x respectively.
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=0)

    # -----------------------------------------------------------
    # Create ranges for the M, N, and K dimensions.
    # These represent the indices that this program instance will compute.
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # -----------------------------------------------------------
    # Initialize accumulator with zeros.
    # The accumulator is of type float32 to maintain precision.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # -----------------------------------------------------------
    # Loop over the K dimension in blocks of BLOCK_SIZE_K.
    # This corresponds to the `for (int k = ...)` loop in the CUDA code.
    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        
        # Calculate pointers to the current tiles of A and B.
        # A is indexed by (m, k) and B is indexed by (k, n).
        a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        
        # Create masks to handle matrices with dimensions that are not multiples
        # of the block sizes. This is a more general approach than the CUDA
        # kernel's coarse `if` check.
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        # Load the tiles of A and B from memory.
        # `other=0.0` pads the tiles with zeros where the mask is false.
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Perform the matrix multiplication on the loaded tiles and accumulate the result.
        # This corresponds to `wmma::mma_sync`.
        accumulator = tl.dot(a, b, accumulator)

    # -----------------------------------------------------------
    # Write back the final result to the C matrix.
    # This corresponds to `wmma::store_matrix_sync`.
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, m: int, k: int, n: int):
    """
    Wrapper function for the Triton matrix multiplication kernel.

    Args:
        A (torch.Tensor): Input matrix A of shape (m, k) and dtype float16.
        B (torch.Tensor): Input matrix B of shape (k, n) and dtype float16.
        C (torch.Tensor): Output matrix C of shape (m, n) and dtype float32.
        m (int): The M dimension of the matrices.
        k (int): The K dimension of the matrices.
        n (int): The N dimension of the matrices.
    """
    # Ensure tensors are on the correct device and have the correct data types.
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on a CUDA device."
    assert A.dtype == torch.float16, "Input tensor A must be of type float16."
    assert B.dtype == torch.float16, "Input tensor B must be of type float16."
    assert C.dtype == torch.float32, "Output tensor C must be of type float32."
    
    # Define block sizes, matching the 16x16x16 WMMA fragment size from the CUDA code.
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16

    # Define the grid for the kernel launch.
    # The grid is 2D, matching the CUDA launch configuration.
    grid = lambda meta: (
        triton.cdiv(n, meta['BLOCK_SIZE_N']),
        triton.cdiv(m, meta['BLOCK_SIZE_M']),
    )

    # Launch the Triton kernel.
    _triton_kernel_impl[grid](
        A, B, C,
        m, k, n,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=4,    # Recommended for performance on modern GPUs
        num_stages=2    # Recommended for performance to enable software pipelining
    )