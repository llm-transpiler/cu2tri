import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel_impl(
    A_ptr,
    C_ptr,
    size1,
    # The inner dimension size (5 in the CUDA code) is passed as a compile-time constant.
    # This allows the Triton compiler to unroll loops and generate more efficient code.
    SIZE2: tl.constexpr,
):
    """
    Triton kernel to perform a row-wise softmax operation.
    This kernel is a direct translation of the provided CUDA logic.
    Each program instance handles one row of the input tensor.
    """
    # Each program instance computes a single row, identified by its program ID.
    row_idx = tl.program_id(0)

    # Boundary check to ensure we don't process rows beyond the specified size1.
    # This is a robust way to handle input sizes that are not multiples of block sizes.
    if row_idx >= size1:
        return

    # Calculate the starting memory addresses for the current row in tensors A and C.
    row_start_ptr_A = A_ptr + row_idx * SIZE2
    row_start_ptr_C = C_ptr + row_idx * SIZE2

    # Create a range of offsets for the columns [0, 1, 2, 3, 4].
    # Triton requires arange range to be power of 2 and constexpr, so use 8 (next power of 2 >= 5)
    # and mask out the extra elements
    col_offsets = tl.arange(0, 8)  # Use constexpr literal 8
    mask = col_offsets < SIZE2  # Only load elements [0,1,2,3,4]

    # Load the 5 float values for the current row from tensor A.
    # Use mask to only load valid elements, set others to 0
    a_row = tl.load(row_start_ptr_A + col_offsets, mask=mask, other=0.0)

    # --- Begin Softmax Computation ---

    # 1. Find the maximum value in the row for numerical stability.
    # Only consider valid elements by setting masked elements to -inf
    masked_a_row = tl.where(mask, a_row, float('-inf'))
    max_val = tl.max(masked_a_row, axis=0)

    # 2. Subtract the max value from each element and exponentiate.
    # Only compute exp for valid elements
    shifted = tl.where(mask, a_row - max_val, 0.0)
    numerator = tl.where(mask, tl.exp(shifted), 0.0)

    # 3. Sum the exponentiated values to get the denominator.
    # Only sum valid elements
    denom = tl.sum(numerator, axis=0)

    # 4. Divide each element by the denominator to normalize.
    softmax_result = tl.where(mask, numerator / denom, 0.0)

    # --- End Softmax Computation ---

    # Store the final computed row back to the output tensor C.
    # Only store valid elements
    tl.store(row_start_ptr_C + col_offsets, softmax_result, mask=mask)


def triton_kernel(A: torch.Tensor, C: torch.Tensor, size1: int, size2: int):
    """
    Wrapper function to launch the Triton softmax kernel.

    This function serves as the entry point, similar to the `cuda_kernel`
    host function in the original CUDA code. It validates inputs and
    configures the grid for the kernel launch.

    Args:
        A (torch.Tensor): Input tensor of shape (size1, size2) and dtype float32.
        C (torch.Tensor): Output tensor of shape (size1, size2) and dtype float32.
        size1 (int): The size of the first dimension of the tensors.
        size2 (int): The size of the second dimension of the tensors.
    """
    # --- Input Validation ---
    # Ensure tensor shapes match the provided dimensions.
    assert A.shape == (size1, size2), f"Input A shape {A.shape} mismatch with ({size1}, {size2})"
    assert C.shape == (size1, size2), f"Output C shape {C.shape} mismatch with ({size1}, {size2})"

    # The original CUDA kernel is hardcoded for an inner dimension of 5.
    assert size2 == 5, "This kernel is specialized for size2=5, matching the CUDA code."

    # Check for correct data types.
    assert A.dtype == torch.float32, "Input tensor A must be of type float32"
    assert C.dtype == torch.float32, "Output tensor C must be of type float32"

    # Triton kernels generally perform best on contiguous tensors.
    assert A.is_contiguous(), "Input tensor A must be contiguous"
    assert C.is_contiguous(), "Output tensor C must be contiguous"

    # Ensure tensors are on the same CUDA device.
    assert A.device == C.device, "Input and output tensors must be on the same device"
    assert A.device.type == 'cuda', "Tensors must be on a CUDA device"

    # --- Grid Configuration ---
    # We launch one Triton program for each row of the input tensor.
    # The grid is 1D with a size equal to the number of rows.
    grid = (size1,)

    # --- Kernel Launch ---
    # Call the JIT-compiled Triton kernel.
    # `size2` is passed as a `constexpr` for compiler optimization.
    _triton_kernel_impl[grid](
        A,
        C,
        size1,
        SIZE2=size2,
    )


if __name__ == '__main__':
    # Example usage and verification
    
    # Define tensor dimensions
    size1 = 25 # An arbitrary number to test, not a multiple of 12
    size2 = 5  # Fixed by the kernel logic

    # Create input tensor on GPU with random data
    A = torch.randn((size1, size2), device='cuda', dtype=torch.float32)
    
    # Create output tensor for the Triton kernel
    C_triton = torch.empty_like(A)

    # Run the Triton kernel
    triton_kernel(A, C_triton, size1, size2)

    # --- Verification using PyTorch's built-in softmax ---
    C_pytorch = torch.nn.functional.softmax(A, dim=1)

    # Compare the results
    # Using a small tolerance for floating-point comparisons
    is_close = torch.allclose(C_triton, C_pytorch, atol=1e-6, rtol=1e-6)
    
    print(f"Input tensor shape: {A.shape}")
    print(f"Output tensor shape: {C_triton.shape}")
    print(f"Verification with torch.softmax: {'SUCCESS' if is_close else 'FAILURE'}")

    # Optional: Print a few rows to inspect
    print("\n--- Triton Output (first 5 rows) ---")
    print(C_triton[:5])
    print("\n--- PyTorch Output (first 5 rows) ---")
    print(C_pytorch[:5])