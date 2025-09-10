import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    stride_q_b, stride_q_s, stride_q_h, stride_q_d,
    stride_k_b, stride_k_s, stride_k_h, stride_k_d,
    stride_v_b, stride_v_s, stride_v_h, stride_v_d,
    stride_o_b, stride_o_s, stride_o_h, stride_o_d,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Triton kernel implementation.
    This kernel computes a custom attention-like operation for each item in a batch and sequence.
    Each program instance handles one (batch, sequence_position) pair.
    """
    # Get program IDs for batch and sequence dimensions, corresponding to CUDA's blockIdx
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    # --- 1. Load Q, K, V for the current (batch, seq_pos) ---
    # The entire (NUM_HEADS, HEAD_DIM) slice for Q, K, and V is loaded into SRAM.
    
    # Create 2D offsets for the heads and head_dim dimensions
    offs_h = tl.arange(0, NUM_HEADS)
    offs_d = tl.arange(0, HEAD_DIM)

    # Create 2D blocks of pointers for Q, K, and V
    q_ptrs = Q_ptr + (pid_b * stride_q_b + pid_s * stride_q_s +
                      offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d)
    k_ptrs = K_ptr + (pid_b * stride_k_b + pid_s * stride_k_s +
                      offs_h[:, None] * stride_k_h + offs_d[None, :] * stride_k_d)
    v_ptrs = V_ptr + (pid_b * stride_v_b + pid_s * stride_v_s +
                      offs_h[:, None] * stride_v_h + offs_d[None, :] * stride_v_d)

    # Load the data into SRAM blocks
    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)

    # --- 2. Compute score = Q @ K.T ---
    # The CUDA code computes score[m, n] = dot(Q[m,:], K[n,:]), which is equivalent to Q @ K.T
    # q: [NUM_HEADS, HEAD_DIM], k: [NUM_HEADS, HEAD_DIM]
    # tl.dot(q, tl.trans(k)) -> [NUM_HEADS, HEAD_DIM] @ [HEAD_DIM, NUM_HEADS] -> [NUM_HEADS, NUM_HEADS]
    score = tl.dot(q, tl.trans(k))

    # --- 3. Apply scaling and softmax ---
    scaling_factor = 1.0 / tl.sqrt(HEAD_DIM.to(tl.float32))
    score = score * scaling_factor

    # Softmax calculation (row-wise on the score matrix)
    # For numerical stability, subtract the max value before exponentiating
    score_max = tl.max(score, axis=1)
    score = score - score_max[:, None]
    score_exp = tl.exp(score.to(tl.float32))
    score_sum = tl.sum(score_exp, axis=1)
    softmax_score = score_exp / score_sum[:, None]
    
    # Ensure the softmax result is in the correct data type for the final dot product
    softmax_score = softmax_score.to(q.dtype)

    # --- 4. Compute output = softmax_score @ V ---
    # softmax_score: [NUM_HEADS, NUM_HEADS], v: [NUM_HEADS, HEAD_DIM]
    # tl.dot(softmax_score, v) -> [NUM_HEADS, HEAD_DIM]
    output_block = tl.dot(softmax_score, v)

    # --- 5. Store the result ---
    # Create a 2D block of output pointers
    output_ptrs = output_ptr + (pid_b * stride_o_b + pid_s * stride_o_s +
                                offs_h[:, None] * stride_o_h + offs_d[None, :] * stride_o_d)
    tl.store(output_ptrs, output_block)


def triton_kernel(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                  output: torch.Tensor, batch_size: int, seq_len: int,
                  num_heads: int, head_dim: int):
    """
    Triton wrapper function with a signature identical to the original CUDA wrapper.

    This function computes a custom attention-like operation where for each item
    in the batch and sequence, Q, K, and V are matrices of shape (num_heads, head_dim).
    The computation is as follows:
    1. score = Q @ K.T
    2. score = score / sqrt(head_dim)
    3. p = softmax(score) (row-wise)
    4. output = p @ V

    Args:
        queries (torch.Tensor): Input tensor Q of shape (B, S, H, D)
        keys (torch.Tensor): Input tensor K of shape (B, S, H, D)
        values (torch.Tensor): Input tensor V of shape (B, S, H, D)
        output (torch.Tensor): Output tensor of shape (B, S, H, D)
        batch_size (int): Batch size (B)
        seq_len (int): Sequence length (S)
        num_heads (int): Number of heads (H)
        head_dim (int): Dimension of each head (D)
    """
    # Check tensor shapes and parameters for correctness
    shape = (batch_size, seq_len, num_heads, head_dim)
    assert queries.shape == shape, f"Expected queries shape {shape}, got {queries.shape}"
    assert keys.shape == shape, f"Expected keys shape {shape}, got {keys.shape}"
    assert values.shape == shape, f"Expected values shape {shape}, got {values.shape}"
    assert output.shape == shape, f"Expected output shape {shape}, got {output.shape}"
    
    # The original CUDA code has hardcoded values for num_heads=6 and head_dim=256.
    # This Triton kernel is specialized for these dimensions.
    assert num_heads == 6, "This kernel is specialized for num_heads=6"
    assert head_dim == 256, "This kernel is specialized for head_dim=256"

    # Grid configuration: launch one program per (batch, seq_len) item.
    # This mirrors the `dim3 grid(batch_size, seq_len)` in the CUDA code.
    grid = (batch_size, seq_len)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        queries, keys, values, output,
        queries.stride(0), queries.stride(1), queries.stride(2), queries.stride(3),
        keys.stride(0), keys.stride(1), keys.stride(2), keys.stride(3),
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
    )


# Example usage and verification code
if __name__ == '__main__':
    # Parameters matching the CUDA code's implicit constants
    batch_size = 4
    seq_len = 1024 # The CUDA code has a hardcoded stride for 2048, but we can use any value here
    num_heads = 6
    head_dim = 256

    # Create random input tensors on the GPU
    torch.manual_seed(0)
    queries = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda', dtype=torch.float32)
    keys = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda', dtype=torch.float32)
    values = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda', dtype=torch.float32)
    
    # Output tensor for the Triton kernel
    output_triton = torch.empty_like(queries)

    print("Running Triton kernel...")
    triton_kernel(queries, keys, values, output_triton, batch_size, seq_len, num_heads, head_dim)
    print("Triton kernel finished.")

    # --- Reference implementation in PyTorch for correctness check ---
    def torch_reference(q, k, v, h_dim):
        # 1. Q @ K.T
        # For each item in batch and sequence, compute (H, D) @ (D, H) -> (H, H)
        scores = torch.matmul(q, k.transpose(-1, -2))
        
        # 2. Scaling
        scaling_factor = 1.0 / (h_dim ** 0.5)
        scores = scores * scaling_factor
        
        # 3. Softmax
        # Applied over the last dimension of the scores tensor (dim=-1)
        softmax_scores = torch.nn.functional.softmax(scores, dim=-1)
        
        # 4. Softmax @ V
        # For each item, compute (H, H) @ (H, D) -> (H, D)
        output = torch.matmul(softmax_scores, v)
        return output

    print("\nRunning PyTorch reference implementation...")
    output_torch = torch_reference(queries, keys, values, head_dim)
    print("PyTorch reference finished.")

    # Compare the results
    print("\n--- Verification ---")
    print("Triton output sample (first 5 elements of one vector):", output_triton[0, 0, 0, :5])
    print("PyTorch output sample (first 5 elements of one vector):", output_torch[0, 0, 0, :5])
    
    # Check for correctness using a reasonable tolerance for floating point math
    is_close = torch.allclose(output_triton, output_torch, atol=1e-5, rtol=1e-5)
    print(f"\nOutputs are close: {is_close}")
    if not is_close:
        # Print more details on failure
        diff = torch.abs(output_triton - output_torch)
        print(f"Maximum difference: {diff.max().item()}")

    # --- Performance Benchmark ---
    # For rigorous results, use triton.testing.do_bench
    print("\n--- Performance Comparison ---")
    
    # Warmup runs
    for _ in range(20):
        triton_kernel(queries, keys, values, output_triton, batch_size, seq_len, num_heads, head_dim)
        torch_reference(queries, keys, values, head_dim)
    
    torch.cuda.synchronize()
    
    # Triton benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(100):
        triton_kernel(queries, keys, values, output_triton, batch_size, seq_len, num_heads, head_dim)
    end_event.record()
    torch.cuda.synchronize()
    triton_time_ms = start_event.elapsed_time(end_event)
    
    # PyTorch benchmark
    start_event.record()
    for _ in range(100):
        output_torch = torch_reference(queries, keys, values, head_dim)
    end_event.record()
    torch.cuda.synchronize()
    torch_time_ms = start_event.elapsed_time(end_event)
    
    print(f"Triton kernel average time: {triton_time_ms / 100:.6f} ms")
    print(f"PyTorch reference average time: {torch_time_ms / 100:.6f} ms")