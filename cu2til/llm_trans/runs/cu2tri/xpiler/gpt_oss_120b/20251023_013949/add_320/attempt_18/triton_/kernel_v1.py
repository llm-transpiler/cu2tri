import torch
import triton
import triton.language as tl

# Compile‑time block size (matches the CUDA launch bounds)
BLOCK_SIZE = 320

@triton.jit(num_warps=10)  # 10 warps × 32 threads = 320 threads per program
def _triton_kernel_impl(
    A_ptr,               # *Pointer* to input A (float32)
    B_ptr,               # *Pointer* to input B (float32)
    C_ptr,               # *Pointer* to output C (float32)
    BLOCK_SIZE: tl.constexpr  # compile‑time constant
):
    """
    Triton kernel that mirrors the CUDA kernel:
    T_add[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x]

    The kernel is launched with many programs (blocks), but each program
    only works on the first BLOCK_SIZE elements, exactly like the original
    CUDA implementation.
    """
    # Thread indices inside the program (0 .. BLOCK_SIZE‑1)
    offs = tl.arange(0, BLOCK_SIZE)

    # Load values. No bounds check – identical to the CUDA version.
    a = tl.load(A_ptr + offs)
    b = tl.load(B_ptr + offs)

    # Compute element‑wise sum.
    c = a + b

    # Store the result back to C.
    tl.store(C_ptr + offs, c)


def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int) -> None:
    """
    Wrapper that launches the Triton kernel with the same interface as the
    original CUDA `cuda_kernel` function.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, GPU, contiguous).
    B : torch.Tensor
        Input tensor B (float32, GPU, contiguous).
    C : torch.Tensor
        Output tensor C (float32, GPU, contiguous). Will be overwritten.
    size : int
        Logical size of the vectors. Used only to compute the grid size,
        mirroring the CUDA launch configuration.
    """
    # Basic sanity checks (mirrors expectations of the CUDA code)
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be float32."
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous."
    assert A.device == B.device == C.device, "All tensors must reside on the same device."
    assert A.device.type == "cuda", "Tensors must be on a CUDA device."

    # Compute the number of programs (blocks) – same formula as CUDA.
    num_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the kernel. BLOCK_SIZE is a compile‑time constant.
    _triton_kernel_impl[grid](
        A, B, C,
        BLOCK_SIZE=BLOCK_SIZE,
        # The launch config (num_warps) is already baked into the @triton.jit decorator.
    )