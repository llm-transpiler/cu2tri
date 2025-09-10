import torch
import triton
import triton.language as tl
import math

# The Triton JIT-compiled kernel.
# It performs an element-wise GELU activation on a 1D tensor.
@triton.jit
def _triton_kernel_impl(
    A_ptr,
    C_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise GELU activation.
    Each program instance processes a block of BLOCK_SIZE elements.
    """
    # 1. Get the program ID for the current instance.
    # This corresponds to the block index in CUDA.
    pid = tl.program_id(axis=0)

    # 2. Calculate the offsets for the elements this program will handle.
    # tl.arange creates a vector [0, 1, ..., BLOCK_SIZE-1].
    # This is equivalent to threadIdx.x in CUDA within a block.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. Create a mask to guard against out-of-bounds memory access.
    # This is necessary because the total size might not be a multiple of BLOCK_SIZE.
    mask = offsets < size

    # 4. Load a block of data from the input tensor A.
    # The mask ensures we only load valid data.
    # 'other=0.0' specifies a default value for out-of-bounds elements,
    # although they won't be used in the final store.
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # 5. Perform the GELU computation.
    # Triton has a built-in GELU function that uses the same tanh approximation
    # as the original CUDA code, which is more efficient and readable.
    # CUDA formula: 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))))
    output = tl.math.gelu(a)

    # 6. Store the result back into the output tensor C.
    # The mask ensures we only write to valid memory locations.
    tl.store(C_ptr + offsets, output, mask=mask)


def triton_kernel(A: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper function for the Triton GELU kernel.

    This function serves as the entry point, similar to the 'cuda_kernel'
    function in the original C++ code. It handles tensor validation,
    grid configuration, and launching the Triton kernel.

    Args:
        A (torch.Tensor): The input tensor of shape (size,).
        C (torch.Tensor): The output tensor of shape (size,).
        size (int): The total number of elements in the tensors.
    """
    # --- Validation Checks ---
    # Ensure tensors are on a CUDA device.
    assert A.is_cuda and C.is_cuda, "Input and output tensors must be on a CUDA device."
    # Ensure data types are float32, matching the CUDA 'float'.
    assert A.dtype == torch.float32, f"Input tensor A must be of type float32, but got {A.dtype}"
    assert C.dtype == torch.float32, f"Output tensor C must be of type float32, but got {C.dtype}"
    # Ensure tensors are contiguous for optimal memory access.
    assert A.is_contiguous(), "Input tensor A must be contiguous"
    assert C.is_contiguous(), "Output tensor C must be contiguous"
    # Verify that the tensor sizes match the provided 'size' parameter.
    assert A.numel() == size, f"Size of tensor A ({A.numel()}) does not match the provided size ({size})"
    assert C.numel() == size, f"Size of tensor C ({C.numel()}) does not match the provided size ({size})"

    # --- Grid Configuration ---
    # Define the size of a processing block. A power of two, like 1024, is
    # generally a good choice for element-wise kernels on modern GPUs.
    # This is a tunable parameter for performance.
    BLOCK_SIZE = 1024

    # Calculate the number of program instances (blocks) needed to cover the entire tensor.
    # triton.cdiv provides a ceiling division to ensure all elements are processed.
    grid = (triton.cdiv(size, BLOCK_SIZE),)

    # --- Kernel Launch ---
    # Launch the Triton kernel with the configured grid.
    # Triton automatically handles passing tensor data pointers to the kernel.
    _triton_kernel_impl[grid](
        A,
        C,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# --- Main execution block for demonstration and verification ---
if __name__ == "__main__":
    # Define the problem size.
    tensor_size = 987654  # Use a non-power-of-two size to test masking.

    # Create input and output tensors on the GPU.
    A_input = torch.randn(tensor_size, dtype=torch.float32, device='cuda')
    C_output_triton = torch.empty_like(A_input)

    print(f"Input tensor created with size: {tensor_size} on device: {A_input.device}")

    # Execute the Triton kernel.
    triton_kernel(A_input, C_output_triton, tensor_size)
    print("Triton kernel executed successfully.")

    # --- Verification ---
    # Compute the reference result using PyTorch's built-in GELU.
    # The 'tanh' approximation is used to match the original CUDA implementation.
    C_output_ref = torch.nn.functional.gelu(A_input, approximate='tanh')

    # Compare the Triton kernel's output with the PyTorch reference.
    # torch.allclose is used for robust floating-point comparisons.
    is_correct = torch.allclose(C_output_triton, C_output_ref, atol=1e-5, rtol=1e-4)

    print(f"Verification against PyTorch successful: {is_correct}")

    if not is_correct:
        print("Verification FAILED!")
        # Find the first differing element to help with debugging.
        diff = torch.abs(C_output_triton - C_output_ref)
        max_diff_val, max_diff_idx = torch.max(diff, 0)
        print(f"Max difference: {max_diff_val.item()} at index {max_diff_idx.item()}")
        print(f"Triton result at index: {C_output_triton[max_diff_idx].item()}")
        print(f"PyTorch ref at index: {C_output_ref[max_diff_idx].item()}")

    assert is_correct, "The Triton kernel output does not match the PyTorch reference."