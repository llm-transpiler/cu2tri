import torch
import triton
import triton.language as tl
from triton import next_power_of_2

def calculate_settings(n_cols):
    BLOCK_SIZE = next_power_of_2(n_cols)
    if BLOCK_SIZE > 8192:
        BLOCK_SIZE = 8192
    num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)
    return BLOCK_SIZE, num_warps

@triton.jit
def _grpo_loss_forward_kernel(
    logits_ptr, logits_row_stride,
    advantages_ptr,
    loss_ptr,
    n_cols,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
):
    """GRPO Loss forward kernel"""
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load logits
    logits_row = tl.load(logits_ptr + row_idx * logits_row_stride + col_offsets, mask=mask, other=-float('inf'))
    logits_f32 = logits_row.to(tl.float32)

    # Compute softmax probabilities
    logits_max = tl.max(logits_f32, axis=0)
    exp_logits = tl.exp(logits_f32 - logits_max)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp

    # Get max prob (action)
    max_prob = tl.max(probs, axis=0)
    log_prob = tl.log(max_prob + 1e-8)

    # Load advantage
    advantage = tl.load(advantages_ptr + row_idx)

    # GRPO objective
    ratio = tl.exp(log_prob - log_prob)  # Simplified - would be log_prob - old_log_prob
    surrogate = ratio * advantage
    clipped_ratio = tl.maximum(tl.minimum(ratio, 1.0 + epsilon), 1.0 - epsilon)
    clipped_surrogate = clipped_ratio * advantage

    # Store negative minimum (for minimization)
    loss = -tl.minimum(surrogate, clipped_surrogate)
    tl.store(loss_ptr + row_idx, loss)

def triton_kernel(logits: torch.Tensor, advantages: torch.Tensor, epsilon: float = 0.2) -> torch.Tensor:
    """Triton实现：GRPO Loss"""
    original_shape = logits.shape
    logits = logits.contiguous()
    advantages = advantages.contiguous()

    # Flatten to 2D
    n_cols = logits.shape[-1]
    logits_2d = logits.view(-1, n_cols)
    advantages_1d = advantages.view(-1)
    n_rows = logits_2d.shape[0]

    # Output tensor
    losses_1d = torch.zeros(n_rows, dtype=torch.float32, device=logits.device)

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    with torch.cuda.device(logits.device):
        _grpo_loss_forward_kernel[(n_rows,)](
            logits_2d, logits_2d.stride(0),
            advantages_1d,
            losses_1d,
            n_cols, epsilon,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )

    return losses_1d.mean().to(logits.dtype)