import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel_impl(
    A_ptr,
    T_sign_ptr,
    size: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise sign operation.
    """
    # Get the program ID for this instance, which corresponds to the block index.
    pid = tl.program_id(axis=0)

    # Calculate the offsets for the elements this program will handle.
    # Each program instance processes a block of BLOCK_SIZE elements.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Create a mask to prevent out-of-bounds memory access for the last block.
    mask = offsets < size

    # Load a block of data from the input tensor A.
    # Apply the mask to avoid reading beyond the tensor's end.
    a = tl.load(A_ptr + offsets, mask=mask)

    # Compute the sign of each element using nested tl.where.
    # This is equivalent to:
    # if a > 0: result = 1.0
    # elif a < 0: result = -1.0
    # else: result = 0.0
    result = tl.where(a > 0.0, 1.0, tl.where(a < 0.0, -1.0, 0.0))

    # Store the resulting block of data into the output tensor T_sign.
    # Apply the mask to avoid writing beyond the tensor's end.
    tl.store(T_sign_ptr + offsets, result, mask=mask)


def triton_kernel(A: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper function for the Triton kernel, providing an entry point with
    functionality identical to the original CUDA code's kernel function.

    Args:
        A (torch.Tensor): The input tensor of type float32.
        C (torch.Tensor): The output tensor to store the sign values, of type float32.
        size (int): The total number of elements in the tensors.
    """
    # Ensure tensors are on the correct device and have the expected properties.
    assert A.is_cuda and C.is_cuda, "Input and output tensors must be on a CUDA device."
    assert A.is_contiguous() and C.is_contiguous(), "Tensors must be contiguous."
    assert A.numel() == size and C.numel() == size, "Tensor sizes must match the 'size' parameter."
    assert A.dtype == torch.float32 and C.dtype == torch.float32, "Tensors must be of type float32."

    # Define the grid for kernel launch.
    # The grid is 1D, and its size is the total number of elements divided by
    # the block size, rounded up.
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)

    # Launch the Triton kernel.
    # A BLOCK_SIZE of 1024 is a good default for element-wise operations.
    _triton_kernel_impl[grid](
        A,
        C,
        size,
        BLOCK_SIZE=1024,
    )


# Example usage and verification script
if __name__ == "__main__":
    # Define the problem size
    size = 1000000

    # Create input tensor 'A' on the GPU with random float32 values
    A = torch.randn(size, device='cuda', dtype=torch.float32)
    # Manually insert some zeros and specific values to test all cases
    A[::10] = 0.0
    A[1] = 5.0
    A[2] = -5.0

    # Create the output tensor 'C' on the GPU
    C_triton = torch.empty(size, device='cuda', dtype=torch.float32)

    print(f"Running Triton kernel for size = {size}...")
    # Call the Triton kernel wrapper
    triton_kernel(A, C_triton, size)

    # --- Verification ---
    print("Verifying correctness against torch.sign...")
    # Use torch.sign as the reference implementation
    C_torch_ref = torch.sign(A)

    # Compare the Triton kernel's output with the reference
    is_correct = torch.allclose(C_triton, C_torch_ref)

    if is_correct:
        print("✅ Triton kernel output is correct.")
    else:
        print("❌ Triton kernel output is incorrect.")
        # Find and print the first mismatch
        mismatched_indices = torch.where(C_triton != C_torch_ref)[0]
        first_mismatch_idx = mismatched_indices[0].item()
        print(f"First mismatch at index {first_mismatch_idx}:")
        print(f"  Input A:      {A[first_mismatch_idx]}")
        print(f"  Triton output: {C_triton[first_mismatch_idx]}")
        print(f"  Torch ref:     {C_torch_ref[first_mismatch_idx]}")

    # Print a few values for manual inspection
    print("\n--- Sample values ---")
    print(f"Input A:      {A[:10].cpu().numpy()}")
    print(f"Triton output: {C_triton[:10].cpu().numpy()}")
    print(f"Torch ref:     {C_torch_ref[:10].cpu().numpy()}")