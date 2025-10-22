import torch
import triton
import triton.language as tl

# Compile‑time block size (matches the CUDA launch bounds)
BLOCK_SIZE = 320

@triton.jit
def _triton_kernel_impl(
    A_ptr,          # *Pointer* to input A (float32)
    B_ptr,          # *Pointer* to input B (float32)
    C_ptr,          # *Pointer* to output C (float32)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that mirrors the CUDA kernel:

        T_add[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x]

    Each Triton program (analogous to a CUDA block) processes BLOCK_SIZE
    elements using threadIdx.x semantics. No global offset is applied,
    matching the original CUDA implementation.
    """
    # Thread indices inside the program (0 .. BLOCK_SIZE‑1)
    offs = tl.arange(0, BLOCK_SIZE)

    # Load values from global memory (no bounds check, identical to CUDA)
    a = tl.load(A_ptr + offs)
    b = tl.load(B_ptr + offs)

    # Compute element‑wise sum
    c = a + b

    # Store the result
    tl.store(C_ptr + offs, c)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int) -> None:
    """
    Wrapper that launches the Triton kernel with the same interface as the
    original CUDA `cuda_kernel` function.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA, contiguous).
    B : torch.Tensor
        Input tensor B (float32, CUDA, contiguous).
    C : torch.Tensor
        Output tensor C (float32, CUDA, contiguous). Will be overwritten.
    size : int
        Logical size of the vectors. Used only to compute the grid size,
        mirroring the CUDA launch configuration.
    """
    # Basic sanity checks (mirroring expectations of the CUDA code)
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("All tensors must be float32.")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise ValueError("All tensors must be contiguous.")
    if A.device != B.device or A.device != C.device:
        raise ValueError("All tensors must reside on the same device.")
    if A.device.type != "cuda":
        raise RuntimeError("Tensors must be on a CUDA device.")
    if size < 0:
        raise ValueError("size must be non‑negative.")
    if size == 0:
        # No work to do; matches CUDA's behavior when launching zero blocks.
        return

    # Compute the number of programs (blocks) – same formula as CUDA.
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the kernel. BLOCK_SIZE is a compile‑time constant.
    # Use 10 warps (10 × 32 = 320 threads) to match the original launch bounds.
    _triton_kernel_impl[grid](
        A, B, C,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=10,
    )