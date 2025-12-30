import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SEQ': 32, 'BLOCK_HEAD': 32}, num_warps=4),
        triton.Config({'BLOCK_SEQ': 64, 'BLOCK_HEAD': 64}, num_warps=8),
        triton.Config({'BLOCK_SEQ': 128, 'BLOCK_HEAD': 64}, num_warps=8),
    ],
    key=['seq_len', 'head_dim'],
)
@triton.jit
def _fused_attn_kernel(
    Q, K, V, Output,
    batch_size, n_heads, seq_len, head_dim, scale,
    BLOCK_SEQ: tl.constexpr, BLOCK_HEAD: tl.constexpr,
):
    """简化的Triton fused attention kernel"""
    # Program ID maps to a specific head within a specific batch
    pid = tl.program_id(0)
    batch_id = pid // n_heads
    head_id = pid % n_heads

    if batch_id >= batch_size or head_id >= n_heads:
        return

    # Compute offsets for this batch and head
    base_offset = (batch_id * n_heads + head_id) * seq_len * head_dim

    # Process the sequence in blocks
    for i in range(0, seq_len, BLOCK_SEQ):
        # Load Q block
        q_row_offsets = i + tl.arange(0, BLOCK_SEQ)
        q_col_offsets = tl.arange(0, BLOCK_HEAD)
        q_mask = (q_row_offsets[:, None] < seq_len) & (q_col_offsets[None, :] < head_dim)

        q_block = tl.load(
            Q + base_offset + q_row_offsets[:, None] * head_dim + q_col_offsets[None, :],
            mask=q_mask
        )

        # Initialize accumulator
        acc = tl.zeros((BLOCK_SEQ, BLOCK_HEAD), dtype=tl.float32)

        # Process attention for all tokens
        for j in range(0, seq_len, BLOCK_SEQ):
            # Load K block
            k_row_offsets = j + tl.arange(0, BLOCK_SEQ)
            k_col_offsets = tl.arange(0, BLOCK_HEAD)
            k_mask = (k_row_offsets[:, None] < seq_len) & (k_col_offsets[None, :] < head_dim)

            k_block = tl.load(
                K + base_offset + k_row_offsets[:, None] * head_dim + k_col_offsets[None, :],
                mask=k_mask
            )

            # Load V block
            v_block = tl.load(
                V + base_offset + k_row_offsets[:, None] * head_dim + k_col_offsets[None, :],
                mask=k_mask
            )

            # Compute attention scores for this block
            # q_block: [BLOCK_SEQ, BLOCK_HEAD], k_block.T: [BLOCK_HEAD, BLOCK_SEQ]
            attn_scores = tl.dot(q_block, k_block.T) * scale

            # Apply softmax on each row
            attn_max = tl.max(attn_scores, axis=1, keep_dims=True)
            attn_exp = tl.exp(attn_scores - attn_max)
            attn_sum = tl.sum(attn_exp, axis=1, keep_dims=True)
            attn_weights = attn_exp / attn_sum

            # Accumulate output: attn_weights [BLOCK_SEQ, BLOCK_SEQ] @ v_block [BLOCK_SEQ, BLOCK_HEAD]
            acc += tl.dot(attn_weights.to(tl.float32), v_block.to(tl.float32))

        # Store output
        tl.store(
            Output + base_offset + q_row_offsets[:, None] * head_dim + q_col_offsets[None, :],
            acc.to(Output.dtype),
            mask=q_mask
        )

def triton_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, output: torch.Tensor,
                  batch_size: int, n_heads: int, seq_len: int, head_dim: int, scale: float) -> torch.Tensor:
    """Triton实现：fused attention (简化版本)"""
    # Ensure inputs are contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = output.contiguous()

    # Launch kernel: one program per head per batch
    grid = (batch_size * n_heads,)
    _fused_attn_kernel[grid](
        q, k, v, output,
        batch_size, n_heads, seq_len, head_dim, scale
    )

    return output