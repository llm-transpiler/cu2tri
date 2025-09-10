import torch
import triton
import triton.language as tl

# The actual Triton kernel (decorated with @triton.jit)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def _triton_kernel_impl(
    A_ptr,
    C_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise ReLU: C = max(A, 0).
    This kernel correctly implements the intended element-wise logic from the CUDA example.
    """
    # Each program instance (similar to a CUDA block) handles a chunk of the data.
    # Get the unique program ID for this instance.
    pid = tl.program_id(axis=0)

    # Calculate the offsets for the data this program will process.
    # This creates a vector of offsets: [0, 1, 2, ..., BLOCK_SIZE-1]
    # and then shifts it by the program's base offset.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Create a boundary-checking mask to prevent out-of-bounds memory access
    # for the last block, which may not be full.
    mask = offsets < n_elements

    # Load a block of data from the input tensor A.
    # The 'mask' argument ensures we don't read past the end of the tensor.
    # 'other=0.0' is a safe fallback for masked-out elements in this specific ReLU operation.
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # Perform the element-wise computation: max(a, 0.0)
    # This operation is vectorized and applied to the entire block 'a'.
    result = tl.maximum(a, 0.0)

    # Store the resulting block of data to the output tensor C.
    # The 'mask' argument ensures we don't write past the end of the tensor.
    tl.store(C_ptr + offsets, result, mask=mask)


# Wrapper function as the entry point with functionality identical to the CUDA code's kernel function
def triton_kernel(A: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper function for launching the Triton ReLU kernel.

    This function has a signature analogous to the original CUDA host function,
    taking tensors and size as input, and handles the grid configuration
    and kernel launch.

    Args:
        A (torch.Tensor): The input tensor (must be float32).
        C (torch.Tensor): The output tensor (must be float32).
        size (int): The total number of elements in the tensors.
    """
    # --- Validation ---
    if not (A.is_cuda and C.is_cuda):
        raise ValueError("Input and output tensors must be on a CUDA device.")
    if not (A.is_contiguous() and C.is_contiguous()):
        raise ValueError("Input and output tensors must be contiguous.")
    if not (A.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("Input and output tensors must be of type float32.")
    if A.numel() != size or C.numel() != size:
        raise ValueError("Tensor number of elements must match the 'size' parameter.")
    if A.shape != C.shape:
        raise ValueError("Input and output tensors must have the same shape.")

    # --- Grid Configuration ---
    # The grid is 1D. Its size is the number of blocks (programs) needed.
    # We use Triton's ceiling division utility to ensure we have enough blocks
    # to cover all 'size' elements.
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)

    # --- Kernel Launch ---
    # The autotuner will select the best BLOCK_SIZE from the defined configs.
    _triton_kernel_impl[grid](
        A_ptr=A,
        C_ptr=C,
        n_elements=size,
    )


# Example usage and verification code to make the script directly executable
if __name__ == "__main__":
    # Check for CUDA device
    if not torch.cuda.is_available():
        print("CUDA device not found. Exiting.")
        exit()

    # --- Setup ---
    # Define the problem size
    tensor_size = 2**20  # 1,048,576 elements

    # Create input tensor with random data on the GPU
    input_tensor = torch.randn(tensor_size, dtype=torch.float32, device='cuda')

    # Create an empty output tensor on the GPU to store the Triton kernel's result
    output_triton = torch.empty_like(input_tensor)

    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    print(f"Problem size: {tensor_size} elements")
    print("-" * 30)

    # --- Execution ---
    # Call the Triton kernel wrapper function
    triton_kernel(input_tensor, output_triton, tensor_size)

    # --- Verification ---
    # Compute the expected result using PyTorch's built-in ReLU for comparison
    output_pytorch = torch.nn.functional.relu(input_tensor)

    # Compare the Triton kernel's output with the PyTorch result
    # torch.allclose is used for safe floating-point comparisons
    is_correct = torch.allclose(output_triton, output_pytorch, atol=1e-5, rtol=1e-5)

    if is_correct:
        print("✅ Triton kernel output matches PyTorch reference output.")
    else:
        print("❌ Triton kernel output does not match PyTorch reference output.")

    # Print a few elements to visually inspect the results
    print("\n--- Sample Outputs (first 10 elements) ---")
    print(f"Input:  {input_tensor[:10].cpu().numpy()}")
    print(f"Triton: {output_triton[:10].cpu().numpy()}")
    print(f"PyTorch:{output_pytorch[:10].cpu().numpy()}")

    # --- Performance Benchmark (optional) ---
    print("\n--- Performance Benchmark ---")
    # Use Triton's built-in benchmarking tool for accurate measurements
    # It handles CUDA synchronization and warm-up automatically
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_kernel(input_tensor, output_triton, tensor_size))
    pytorch_ms, _, _ = triton.testing.do_bench(lambda: torch.nn.functional.relu(input_tensor))

    # Calculate bandwidth in GB/s
    # For ReLU, we read and write the tensor once (2 * size * bytes_per_element)
    gb_per_s = lambda ms: 2 * tensor_size * input_tensor.element_size() / (ms * 1e-3) / 1e9

    print(f"Triton kernel: {ms:.4f} ms | Bandwidth: {gb_per_s(ms):.2f} GB/s")
    print(f"PyTorch kernel:  {pytorch_ms:.4f} ms | Bandwidth: {gb_per_s(pytorch_ms):.2f} GB/s")