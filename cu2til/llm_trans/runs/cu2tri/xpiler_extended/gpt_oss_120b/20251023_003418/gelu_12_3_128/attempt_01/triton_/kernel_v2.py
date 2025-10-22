import torch
import triton
import triton.language as tl
import math

# Pre‑computed constant sqrt(2 / pi)
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


@triton.jit
def _triton_kernel_impl(
    A_ptr: tl.pointer(tl.float32),  # float* (input)
    C_ptr: tl.pointer(tl.float32),  # float* (output)
    size: tl.int32,                 # total number of elements to process
    BLOCK_SIZE: tl.constexpr,       # compile‑time block size (1024)
):
    """
    Triton implementation of the CUDA gelu kernel.
    """
    pid = tl.program_id(0)  # block index (1‑D grid)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # global indices
    mask = offsets < size  # bounds check

    # Load input values (masked)
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # Gelu approximation:
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))
    x = a
    x3 = x * x * x
    inner = SQRT_2_OVER_PI * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.tanh(inner))

    # Store result (masked)
    tl.store(C_ptr + offsets, gelu, mask=mask)


def triton_kernel(A: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that mimics the original CUDA kernel signature:
        void cuda_kernel(float *A, float *C, int size)
    """
    assert A.is_cuda and C.is_cuda, "A and C must be CUDA tensors"
    assert A.dtype == torch.float32 and C.dtype == torch.float32, "Tensors must be float32"
    BLOCK_SIZE = 1024
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A.data_ptr(),
        C.data_ptr(),
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# Example usage / verification (can be removed in production)
if __name__ == "__main__":
    torch.manual_seed(0)
    N = 4608
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    triton_kernel(A, C, N)

    # Reference implementation in PyTorch
    def gelu_ref(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

    C_ref = gelu_ref(A)
    torch.testing.assert_allclose(C, C_ref, atol=1e-6, rtol=1e-5)
    print("Triton kernel verification passed.")