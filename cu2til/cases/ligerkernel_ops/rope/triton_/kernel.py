import torch
import triton
import triton.language as tl

from triton import next_power_of_2

def calculate_settings(n_cols):
    """Calculate optimal block size and warps based on tensor dimensions"""
    BLOCK_SIZE = next_power_of_2(n_cols)
    if BLOCK_SIZE > 8192:
        BLOCK_SIZE = 8192
    num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)
    return BLOCK_SIZE, num_warps

def precompute_freqs_cis_triton(dim: int, seq_len: int, theta: float = 10000.0, device='cuda') -> torch.Tensor:
    """Precompute frequency tensor for complex exponentials (cos, sin)"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

@triton.jit
def _rope_kernel(
    q_ptr, q_row_stride,
    k_ptr, k_row_stride,
    cos_ptr, cos_row_stride,
    sin_ptr, sin_row_stride,
    seq_len,
    n_q_heads, n_k_heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RoPE kernel that applies rotary position embeddings to query and key tensors
    """
    # Get program ID representing batch and sequence position
    program_id = tl.program_id(0).to(tl.int64)
    batch_idx = program_id // seq_len
    seq_idx = program_id % seq_len

    # Load cos and sin for this sequence position
    cos_offsets = tl.arange(0, BLOCK_SIZE)
    cos_mask = cos_offsets < head_dim // 2
    cos_row = tl.load(cos_ptr + seq_idx * cos_row_stride + cos_offsets, mask=cos_mask, other=0.0)
    sin_row = tl.load(sin_ptr + seq_idx * sin_row_stride + cos_offsets, mask=cos_mask, other=0.0)

    # Process each head
    for head_idx in range(n_q_heads):
        # Calculate base offset for this head and sequence
        q_base_offset = (batch_idx * seq_len * n_q_heads + seq_idx * n_q_heads + head_idx) * head_dim

        # Load first and second halves of query
        q_offsets_1 = q_base_offset + tl.arange(0, head_dim // 2)
        q_offsets_2 = q_base_offset + head_dim // 2 + tl.arange(0, head_dim // 2)
        q_mask_1 = tl.arange(0, head_dim // 2) < head_dim // 2
        q_mask_2 = tl.arange(0, head_dim // 2) < head_dim // 2

        q1 = tl.load(q_ptr + q_offsets_1, mask=q_mask_1, other=0.0).to(cos_row.dtype)
        q2 = tl.load(q_ptr + q_offsets_2, mask=q_mask_2, other=0.0).to(cos_row.dtype)

        # Apply rotation: q' = [q1*cos - q2*sin, q2*cos + q1*sin]
        new_q1 = q1 * cos_row - q2 * sin_row
        new_q2 = q2 * cos_row + q1 * sin_row

        # Store back
        tl.store(q_ptr + q_offsets_1, new_q1, mask=q_mask_1)
        tl.store(q_ptr + q_offsets_2, new_q2, mask=q_mask_2)

    # Process keys (might have different number of heads)
    for head_idx in range(n_k_heads):
        # Calculate base offset for this head and sequence
        k_base_offset = (batch_idx * seq_len * n_k_heads + seq_idx * n_k_heads + head_idx) * head_dim

        # Load first and second halves of key
        k_offsets_1 = k_base_offset + tl.arange(0, head_dim // 2)
        k_offsets_2 = k_base_offset + head_dim // 2 + tl.arange(0, head_dim // 2)
        k_mask_1 = tl.arange(0, head_dim // 2) < head_dim // 2
        k_mask_2 = tl.arange(0, head_dim // 2) < head_dim // 2

        k1 = tl.load(k_ptr + k_offsets_1, mask=k_mask_1, other=0.0).to(cos_row.dtype)
        k2 = tl.load(k_ptr + k_offsets_2, mask=k_mask_2, other=0.0).to(cos_row.dtype)

        # Apply rotation: k' = [k1*cos - k2*sin, k2*cos + k1*sin]
        new_k1 = k1 * cos_row - k2 * sin_row
        new_k2 = k2 * cos_row + k1 * sin_row

        # Store back
        tl.store(k_ptr + k_offsets_1, new_k1, mask=k_mask_1)
        tl.store(k_ptr + k_offsets_2, new_k2, mask=k_mask_2)

def triton_kernel(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton实现：RoPE (Rotary Position Embedding)
    高效的旋转位置编码实现
    """
    batch_size, seq_len, n_q_heads, head_dim = q.shape
    _, _, n_k_heads, _ = k.shape

    # Ensure contiguous inputs
    q = q.contiguous()
    k = k.contiguous()

    # Create output tensors (modify in-place)
    q_out = q.clone()
    k_out = k.clone()

    # Precompute cos and sin values
    max_seq_len = max(seq_len, k.shape[1])
    freqs_cos, freqs_sin = precompute_freqs_cis_triton(head_dim, max_seq_len, device=q.device)

    # Calculate optimal block size
    BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)

    # Grid configuration: batch_size * seq_len programs
    grid = (batch_size * seq_len,)

    # Launch kernel
    with torch.cuda.device(q.device):
        _rope_kernel[grid](
            q_out,
            q_out.stride(1) * n_q_heads,  # seq_len stride
            k_out,
            k_out.stride(1) * n_k_heads,  # seq_len stride
            freqs_cos,
            freqs_cos.stride(0),  # seq_len stride
            freqs_sin,
            freqs_sin.stride(0),  # seq_len stride
            seq_len,
            n_q_heads,
            n_k_heads,
            head_dim,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return q_out, k_out