import torch
import triton
import triton.language as tl

# Triton kernel implementing vector addition with a hardcoded bound of 2304 elements.
@triton.jit
def _triton_kernel_impl(A, B, T_add, BLOCK_SIZE: tl.constexpr):
    # Program ID identifies the block (equivalent to blockIdx.x in CUDA)
    pid = tl.program_id(0)
    # Compute the global offset for each thread in this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Apply the same guard as the original CUDA kernel: only process indices < 2304
    mask = offsets < 2304
    # Load values from A and B (masked loads return 0.0 for out‑of‑range elements)
    =(A + offsets, mask=mask, other=0.)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    # Store the sum into T_add (masked store)
    tl.store(T_add + offsets, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry‑point that mimics the original CUDA wrapper signature:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A, B, C : torch.Tensor
        1‑D tensors of dtype torch.float32 residing on the CUDA device.
    size : int
        Logical size of the vectors (used only to compute grid dimensions,
        mirroring the CUDA launch configuration).
    """
    # Ensure inputs are on the GPU and have the expected dtype.
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be of type torch.float32"
    # Block size matches the __launch_bounds__(1024) in the CUDA kernel.
    BLOCK_SIZE = 1024
    # Compute the number of blocks needed to cover `size` elements (rounded up).
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the Triton kernel.
    _triton_kernel_impl[(num_blocks,)](A, B, C, BLOCK_SIZE=BLOCK_SIZE)