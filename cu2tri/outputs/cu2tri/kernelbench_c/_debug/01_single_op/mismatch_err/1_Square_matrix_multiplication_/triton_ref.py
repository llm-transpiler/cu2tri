import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------
# Triton Kernel for Matrix Multiplication
# ----------------------------------------------------------------

@triton.autotune(
    configs=[
        # Basic configurations for various tile sizes
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}),
        
        # Configurations with a larger K tile size, good for K-bound problems
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}),
        
        # Configuration resembling the original CUDA code's 32x32 tile
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 4}),
        
        # High-performance configurations for modern GPUs like NVIDIA Ada Lovelace
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Tunable parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton kernel for computing matrix multiplication C = A @ B.

    This kernel uses tiling and grouping to achieve high performance.
    - Tiling: The M, N, and K dimensions are partitioned into blocks (tiles).
      Each instance of this kernel (a "program") computes one M x N tile of the output C.
    - Grouping: `GROUP_SIZE_M` programs are grouped together to improve L2 cache reuse
      for the A matrix, which can provide a significant speedup.
    - `tl.dot`: The core computation is performed using `tl.dot`, which leverages
      tensor cores on supported hardware for maximum throughput.
    - Masking: Boundary conditions for matrices with dimensions not divisible by
      the block sizes are handled using masks.
    """
    # -----------------------------------------------------------
    # Map program ids to M and N dimensions
    # -----------------------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    
    # Create a 2D grid of program ids, grouped along the M dimension
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    
    # Program's position within the group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # Final program ids for the M and N dimensions
    pid_m = first_pid_m + (pid % group_size)
    pid_n = (pid // group_size) % num_pid_n

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B
    # ----------------------------------------------------------
    # Offsets for the M, N, and K dimensions for this program
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers to the top-left corner of the first tiles of A and B
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Initialize accumulator with zeros
    # -----------------------------------------------------------
    # `accumulator` is a `BLOCK_SIZE_M x BLOCK_SIZE_N` tile held in registers.
    # Use tl.float32 for accumulation to maintain precision.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # -----------------------------------------------------------
    # Loop over the K dimension
    # -----------------------------------------------------------
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next tile of A and B from global memory.
        # Masking is used to handle boundary conditions.
        k_tile_indices = k * BLOCK_SIZE_K + offs_k
        
        a_mask = (offs_m[:, None] < M) & (k_tile_indices[None, :] < K)
        b_mask = (k_tile_indices[:, None] < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Perform the matrix multiplication for the current tiles and accumulate the result.
        # `tl.dot` is a high-performance operation that utilizes tensor cores.
        accumulator += tl.dot(a, b)
        
        # Advance the pointers to the next K-tile
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Cast the accumulator to the output tensor's data type
    c = accumulator.to(C.dtype.element_ty)

    # -----------------------------------------------------------
    # Write the result tile to global memory
    # -----------------------------------------------------------
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Create a mask to avoid writing out of bounds
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def forward(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes matrix multiplication C = A @ B using a high-performance Triton kernel.

    This function serves as a wrapper that handles input validation, tensor allocation,
    grid definition, and the launch of the Triton kernel. It is designed to be a
    drop-in replacement for torch.matmul for contiguous float32 tensors on CUDA.

    Args:
        A (torch.Tensor): The first input tensor of shape (M, K). Must be a contiguous
                          float32 tensor on a CUDA device.
        B (torch.Tensor): The second input tensor of shape (K, N). Must be a contiguous
                          float32 tensor on a CUDA device.

    Returns:
        torch.Tensor: The output tensor C of shape (M, N).
    """
    # 1. Input Validation
    # The original CUDA code was for (N, N) x (N, N) float32 matrices.
    # This implementation is generalized to (M, K) x (K, N) while maintaining
    # the data type and contiguity requirements.
    assert A.is_cuda and B.is_cuda, "Input tensors must be on a CUDA device"
    assert A.is_contiguous() and B.is_contiguous(), "Input tensors must be contiguous"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Input tensors must be of type float32"
    assert A.shape[1] == B.shape[0], f"Incompatible dimensions for matmul: A.shape={A.shape}, B.shape={B.shape}"

    # 2. Get matrix dimensions
    M, K = A.shape
    _, N = B.shape

    # 3. Allocate the output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # 4. Define the grid for launching the kernel
    # The grid is 1D, and its size is the total number of blocks needed.
    # The kernel internally maps this 1D grid to a 2D grid of blocks.
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )

    # 5. Launch the Triton kernel
    _matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )

    return C
