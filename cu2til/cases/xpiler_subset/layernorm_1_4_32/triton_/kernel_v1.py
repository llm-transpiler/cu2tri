import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Triton Kernel Implementation
# -----------------------------------------------------------------------------

@triton.jit
def _triton_kernel_impl(
    A_ptr, gamma_ptr, beta_ptr, B_ptr,
    N_ROWS: tl.int32,
    D_MODEL: tl.int32,
    stride_a_row: tl.int32,
    stride_b_row: tl.int32,
    BLOCK_SIZE_D: tl.constexpr,
    EPSILON: tl.constexpr,
):
    """
    Triton kernel for Layer Normalization.
    This kernel processes one row of the input tensor per program instance.
    It computes the mean and variance for the row, normalizes it,
    then applies a scale (gamma) and shift (beta).

    The computation follows the formula:
    B = ((A - mean(A)) * gamma) / (sqrt(variance(A)) + epsilon) + beta
    This is mathematically equivalent to the standard LayerNorm formula and
    matches the (somewhat unusual) order of operations in the provided CUDA code.
    """
    # 1. Get the ID of the row this program is responsible for.
    row_idx = tl.program_id(axis=0)

    # 2. Guard against out-of-bounds access for the last block.
    if row_idx >= N_ROWS:
        return

    # 3. Define offsets for loading a full row of D_MODEL elements.
    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < D_MODEL

    # 4. Load the input row from A, and the full gamma and beta vectors.
    #    Padded values in 'a' are set to 0.0 to not affect the sum.
    a_ptr = A_ptr + row_idx * stride_a_row
    a = tl.load(a_ptr + d_offsets, mask=d_mask, other=0.0)

    gamma = tl.load(gamma_ptr + d_offsets, mask=d_mask)
    beta = tl.load(beta_ptr + d_offsets, mask=d_mask)

    # 5. Compute mean of the row.
    #    The sum is performed over the loaded block 'a', and then divided
    #    by the actual dimension size D_MODEL.
    mean = tl.sum(a, axis=0) / D_MODEL

    # 6. Compute variance of the row.
    #    We use tl.where to ensure that padded elements do not contribute to the variance.
    a_minus_mean = tl.where(d_mask, a - mean, 0.0)
    variance = tl.sum(a_minus_mean * a_minus_mean, axis=0) / D_MODEL
    
    # 7. Compute standard deviation.
    std_dev = tl.sqrt(variance)

    # 8. Normalize, scale, and shift.
    #    The CUDA code's order of operations is:
    #    diff = (A - mean) * gamma
    #    diff = diff / (std_dev + epsilon)
    #    B = diff + beta
    #    This is equivalent to the fused operation below.
    a_normalized = (a - mean) / (std_dev + EPSILON)
    b = a_normalized * gamma + beta

    # 9. Store the final result to B.
    b_ptr = B_ptr + row_idx * stride_b_row
    tl.store(b_ptr + d_offsets, b, mask=d_mask)


# -----------------------------------------------------------------------------
# Wrapper Function
# -----------------------------------------------------------------------------

def triton_kernel(A: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, B: torch.Tensor,
                  batch_size: int, seq_length: int, d_model: int):
    """
    Wrapper function for the Layer Normalization Triton kernel.

    Args:
        A (torch.Tensor): Input tensor of shape (batch_size * seq_length, d_model).
        gamma (torch.Tensor): Scaling tensor (weight) of shape (d_model,).
        beta (torch.Tensor): Shifting tensor (bias) of shape (d_model,).
        B (torch.Tensor): Output tensor of shape (batch_size * seq_length, d_model).
        batch_size (int): Batch size dimension.
        seq_length (int): Sequence length dimension.
        d_model (int): Feature dimension (d_model).
    """
    # Ensure all tensors are on the same CUDA device and have the correct dtype
    assert all(t.is_cuda for t in [A, gamma, beta, B]), "All tensors must be on a CUDA device."
    assert all(t.dtype == torch.float32 for t in [A, gamma, beta, B]), "All tensors must be of type float32."

    # The total number of rows to process
    N_ROWS = batch_size * seq_length

    # Reshape A and B to 2D if they are not already, for stride calculation
    # This is a safe view that doesn't copy data
    A_2d = A.view(N_ROWS, d_model)
    B_2d = B.view(N_ROWS, d_model)

    # Define the grid for the kernel launch.
    # We launch one program per row.
    grid = (N_ROWS,)

    # Choose the smallest power of 2 greater than or equal to d_model for BLOCK_SIZE_D.
    # This helps Triton generate more efficient code. The kernel handles the exact
    # size using masking.
    BLOCK_SIZE_D = triton.next_power_of_2(d_model)

    # Launch the kernel
    _triton_kernel_impl[grid](
        A_2d, gamma, beta, B_2d,
        N_ROWS,
        d_model,
        A_2d.stride(0),
        B_2d.stride(0),
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        EPSILON=1e-5,
    )


# -----------------------------------------------------------------------------
# Main execution block for testing and verification
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Test Configuration ---
    batch_size = 2
    seq_length = 2
    d_model = 32
    
    # The CUDA kernel hardcodes idx < 4, which corresponds to N_ROWS = 4
    # and d_model = 32. We use these values for a direct comparison.
    N_ROWS = batch_size * seq_length

    print(f"Testing with: batch_size={batch_size}, seq_length={seq_length}, d_model={d_model}")
    print(f"Total rows (N_ROWS): {N_ROWS}")

    # --- Reference PyTorch Implementation ---
    def reference_pytorch(A: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        A PyTorch implementation that exactly matches the logic of the CUDA kernel for verification.
        """
        # Reshape for row-wise operations
        A_2d = A.view(N_ROWS, d_model)
        
        # Calculate mean and variance per row
        mean = A_2d.mean(dim=-1, keepdim=True)
        # Use unbiased=False to match the CUDA kernel's (sum / N) logic
        var = A_2d.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var)
        
        # Apply the normalization, scaling, and shifting
        # The formula is derived from a careful trace of the CUDA kernel's operations
        a_minus_mean = A_2d - mean
        normalized = a_minus_mean / (std + eps)
        
        return normalized * gamma + beta

    # --- Data Initialization ---
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA device not found. Exiting.")
        exit()

    # Create input tensors on the GPU
    A = torch.randn((N_ROWS, d_model), device=device, dtype=torch.float32)
    gamma = torch.randn(d_model, device=device, dtype=torch.float32)
    beta = torch.randn(d_model, device=device, dtype=torch.float32)
    
    # Create output tensor for the Triton kernel
    B_triton = torch.empty_like(A)

    # --- Kernel Execution and Verification ---
    print("\nRunning Triton kernel...")
    triton_kernel(A, gamma, beta, B_triton, batch_size, seq_length, d_model)

    print("Running reference PyTorch implementation...")
    B_pytorch = reference_pytorch(A, gamma, beta)

    # --- Comparison ---
    print("Comparing results...")
    # Use a slightly higher tolerance for floating point comparisons
    are_close = torch.allclose(B_triton, B_pytorch, atol=1e-5, rtol=1e-4)
    
    if are_close:
        print("✅ Pass: Triton kernel output matches the reference PyTorch implementation.")
    else:
        print("❌ Fail: Triton kernel output does NOT match the reference PyTorch implementation.")
        # Print detailed differences for debugging
        diff = torch.abs(B_triton - B_pytorch)
        print(f"   - Max absolute difference: {diff.max().item()}")
        print(f"   - Max relative difference: {(diff / torch.abs(B_pytorch)).max().item()}")

    # --- Performance Benchmark (Optional) ---
    print("\nRunning simple benchmark...")
    try:
        # Use Triton's built-in benchmarking tool
        ms, max_ms, min_ms = triton.testing.do_bench(lambda: triton_kernel(A, gamma, beta, B_triton, batch_size, seq_length, d_model))
        print(f"Triton kernel execution time: {ms:.4f} ms (min: {min_ms:.4f} ms, max: {max_ms:.4f} ms)")

        ms_ref, _, _ = triton.testing.do_bench(lambda: reference_pytorch(A, gamma, beta))
        print(f"PyTorch reference execution time: {ms_ref:.4f} ms")
    except Exception as e:
        print(f"Could not run benchmark: {e}")