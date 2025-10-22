import torch
import triton
import triton.language as tl

# Triton kernel implementing the elementwise add with the same block/grid
# tiling strategy as the original CUDA kernel.
#
# - Name must be _triton_kernel_impl (per requirements)
# - BLOCK is a compile-time constant (tl.constexpr)
# - The kernel processes a vector of length BLOCK per program (program_id(0) ~ blockIdx.x)
# - It loops 8 times (matching the outer loop in the CUDA kernel)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_ptr, N, BLOCK: tl.constexpr):
    """
    A_ptr, B_ptr, T_ptr: pointers (torch tensors) to float32 arrays on GPU
    N: integer (number of valid elements). This mirrors the wrapper 'size'.
    BLOCK: tile size (constexpr), set from the launcher.
    """
    # program id corresponds to CUDA blockIdx.x (we will launch num_blocks programs)
    pid = tl.program_id(0)

    # local thread-like range within a block (0 .. BLOCK-1)
    local_idx = tl.arange(0, BLOCK)

    # base offsets for this "block" (mimics blockIdx.x * 1024 + threadIdx.x)
    base_offsets = pid * BLOCK + local_idx  # shape: (BLOCK,)

    # stride between outer tiles: grid_dim * block_dim = 256 * BLOCK
    # In the original CUDA kernel stride is 262144 (256 * 1024). We compute it
    # via the same relation so varying BLOCK still works if needed.
    stride = 256 * BLOCK

    # Loop 8 times as in the original CUDA kernel:
    # index = outer * 262144 + blockIdx.x * 1024 + threadIdx.x
    for outer in range(8):
        idx = outer * stride + base_offsets  # vector of indices
        mask = idx < N  # valid positions within bounds

        # guarded loads/stores to avoid OOB accesses
        a = tl.load(A_ptr + idx, mask=mask, other=0.0)
        b = tl.load(B_ptr + idx, mask=mask, other=0.0)
        c = a + b
        tl.store(T_ptr + idx, c, mask=mask)


# Wrapper that mirrors the original host wrapper signature:
# extern "C" void cuda_kernel(float *A, float *B, float *C, int size)
def triton_kernel(A, B, C, size):
    """
    Entry point that configures the grid and launches the Triton kernel.

    Parameters:
    - A, B, C: torch.Tensor on CUDA device with dtype=torch.float32
    - size: int (number of valid elements). This will be passed to the kernel
            as the bound for masking (so behavior matches for size==2048000).
    """
    # Basic checks to ensure types and devices match expected CUDA behaviour
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor instances")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors (on GPU)")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must have dtype torch.float32")
    if A.numel() < 1 or B.numel() < 1 or C.numel() < 1:
        raise ValueError("A, B, C must be non-empty tensors")

    # The original CUDA host wrapper used:
    #   dim3 blockSize(1024);
    #   dim3 numBlocks(256);
    # So we replicate these exact launch dimensions.
    block_size = 1024
    num_blocks = 256

    # Ensure provided 'size' is an int and reasonable
    N = int(size)

    # Optional safety: ensure the tensors have at least N elements to avoid UB.
    # The original CUDA kernel used a hard limit 2048000 inside the kernel;
    # here we use the passed 'size' to determine valid bounds (more flexible).
    if A.numel() < N or B.numel() < N or C.numel() < N:
        raise ValueError("A, B, C must each contain at least 'size' elements")

    # Launch the Triton kernel with the requested grid.
    # BLOCK is provided as a compile-time constant.
    _triton_kernel_impl[(num_blocks,)](A, B, C, N, BLOCK=block_size)