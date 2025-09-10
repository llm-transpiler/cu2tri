import torch
import triton
import triton.language as tl

# The Triton kernel, equivalent to the CUDA __global__ function.
@triton.jit
def _triton_kernel_impl(
    input_ptr,
    kernel_ptr,
    output_ptr,
    input_size: int,
    output_size: int,
    # Use tl.constexpr for values known at compile time for better optimization.
    # The CUDA kernel has a hardcoded loop for j < 3.
    KERNEL_SIZE: tl.constexpr,
    # The CUDA kernel uses a block size of 5. We match this for equivalence.
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Triton implementation of the 1D convolution kernel.
    Each program instance computes a block of outputs.
    """
    # Get the program ID for the current program instance. This is equivalent to blockIdx.x.
    pid = tl.program_id(axis=0)

    # Create a vector of offsets for the threads within this program instance.
    # This is equivalent to threadIdx.x.
    # tl.arange(0, BLOCK_SIZE_M) -> [0, 1, 2, 3, 4]
    offsets_m = tl.arange(0, BLOCK_SIZE_M)
    
    # Calculate the global indices for the output elements this program will handle.
    # This is equivalent to `blockIdx.x * blockDim.x + threadIdx.x`.
    idx = pid * BLOCK_SIZE_M + offsets_m

    # Create a mask to prevent out-of-bounds memory operations.
    # This is equivalent to the `if (idx < 5)` check in the CUDA kernel,
    # where output_size is 5.
    output_mask = idx < output_size

    # Initialize an accumulator vector with zeros for each thread.
    # The shape (BLOCK_SIZE_M,) corresponds to the number of threads in this program.
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # The inner loop for the convolution.
    # Since KERNEL_SIZE is a tl.constexpr, the Triton compiler will unroll this loop.
    for j in range(KERNEL_SIZE):
        # Calculate the offsets to read from the input tensor.
        input_offsets = idx + j
        
        # Create a mask to ensure we only read within the bounds of the input tensor.
        # We combine the output_mask to avoid doing work for padding threads.
        input_mask = output_mask & (input_offsets < input_size)

        # Load a vector of input values and a single kernel value.
        # `other=0.0` ensures that out-of-bounds reads do not cause errors and contribute zero,
        # which is correct for convolution padding.
        input_val = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        kernel_val = tl.load(kernel_ptr + j)

        # Perform the element-wise multiply-add operation.
        accumulator += input_val * kernel_val

    # Store the final computed values into the output tensor.
    # The output_mask ensures that only threads corresponding to valid output indices write their results.
    tl.store(output_ptr + idx, accumulator, mask=output_mask)


def triton_kernel(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, 
                  input_size: int, output_size: int):
    """
    Wrapper function for the Triton kernel, providing an interface identical
    to the original CUDA wrapper.

    Args:
        input (torch.Tensor): The 1D input tensor. Must be on CUDA device.
        kernel (torch.Tensor): The 1D kernel tensor. Must be on CUDA device.
        output (torch.Tensor): The 1D output tensor. Must be on CUDA device.
        input_size (int): The number of elements in the input tensor.
        output_size (int): The number of elements in the output tensor.
    """
    # Basic validation for tensor properties.
    assert all(t.is_cuda for t in [input, kernel, output]), "All tensors must be on a CUDA device."
    assert all(t.dtype == torch.float32 for t in [input, kernel, output]), "All tensors must be of type float32."

    # Define constants based on the original CUDA code's logic.
    # The CUDA code uses a fixed block size of 5.
    BLOCK_SIZE_M = 5
    # The CUDA code has a fixed kernel size of 3 in its loop.
    KERNEL_SIZE = 3
    assert kernel.numel() >= KERNEL_SIZE, f"Kernel tensor must have at least {KERNEL_SIZE} elements."

    # Define the grid for the kernel launch.
    # The grid is 1D. The number of programs (blocks) is calculated by dividing
    # the total output size by the block size, rounding up.
    grid = lambda meta: (triton.cdiv(output_size, meta['BLOCK_SIZE_M']),)

    # Launch the Triton kernel.
    _triton_kernel_impl[grid](
        input,
        kernel,
        output,
        input_size,
        output_size,
        KERNEL_SIZE=KERNEL_SIZE,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )


# Example usage and verification block.
if __name__ == "__main__":
    # --- Setup based on the CUDA code's logic ---
    # The CUDA kernel is hardcoded to process 5 output elements.
    output_size = 5
    kernel_size = 3
    
    # To avoid out-of-bounds access, the input size must be at least
    # output_size + kernel_size - 1.
    # For output_idx=4 and kernel_idx=2, we access input[4+2]=input[6].
    # So, the input tensor needs at least 7 elements.
    input_size = 7

    # --- Create Tensors ---
    torch.manual_seed(0)
    # Create tensors on the GPU with float32 data type.
    input_tensor = torch.randn(input_size, device='cuda', dtype=torch.float32)
    kernel_tensor = torch.randn(kernel_size, device='cuda', dtype=torch.float32)
    # Output tensor to be populated by the Triton kernel.
    output_triton = torch.empty(output_size, device='cuda', dtype=torch.float32)

    print("Input Tensor:", input_tensor)
    print("Kernel Tensor:", kernel_tensor)

    # --- Launch Triton Kernel ---
    triton_kernel(input_tensor, kernel_tensor, output_triton, input_size, output_size)

    # --- Verification ---
    # Compute the reference solution on the CPU for comparison.
    output_ref = torch.zeros(output_size, dtype=torch.float32)
    input_cpu = input_tensor.cpu()
    kernel_cpu = kernel_tensor.cpu()

    for i in range(output_size):
        for j in range(kernel_size):
            output_ref[i] += input_cpu[i + j] * kernel_cpu[j]
            
    print("\nTriton Output:", output_triton)
    print("Reference Output:", output_ref.cuda())

    # Compare the Triton kernel's output with the reference solution.
    # torch.allclose is used to handle potential floating-point inaccuracies.
    if torch.allclose(output_triton, output_ref.cuda(), atol=1e-6, rtol=1e-6):
        print("\n✅ Verification successful: Triton output matches the reference output.")
    else:
        print("\n❌ Verification failed: Triton output does not match the reference output.")
        print("Difference:", torch.abs(output_triton - output_ref.cuda()))