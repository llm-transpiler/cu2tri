import torch
import triton
import triton.language as tl

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576
# However, setting limit as 65536 is faster because of less register spilling
MAX_FUSED_SIZE = 65536 // 2

@triton.jit
def _cross_entropy_forward_kernel(
    X_ptr, X_stride,
    Y_ptr, Y_stride,
    loss_ptr, loss_stride,
    n_cols,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Cross Entropy forward kernel
    Computes cross entropy loss for each row
    """
    row_idx = tl.program_id(0)

    # Get row pointers
    X_row_ptr = X_ptr + row_idx * X_stride
    Y_row_ptr = Y_ptr + row_idx * Y_stride
    loss_ptr = loss_ptr + row_idx * loss_stride

    # Load target
    target = tl.load(Y_row_ptr)

    # Skip if target is ignore_index
    if target == ignore_index:
        tl.store(loss_ptr, 0.0)
        return

    # Load logits row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    logits = tl.load(X_row_ptr + col_offsets, mask=mask, other=float('-inf')).to(tl.float32)

    # Compute max for numerical stability
    logits_max = tl.max(logits, axis=0)
    logits = logits - logits_max

    # Compute exp and sum
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)

    # Get logit for target
    target_logit = tl.load(X_row_ptr + target)

    # Compute cross entropy loss
    loss = tl.log(sum_exp) - (target_logit - logits_max)

    tl.store(loss_ptr, loss)

def triton_kernel(_input: torch.Tensor, weight: torch.Tensor, target: torch.Tensor,
                 bias: torch.Tensor = None, ignore_index: int = -100, reduction: str = "mean") -> torch.Tensor:
    """
    Triton实现：Fused Linear Cross Entropy
    融合线性层和交叉熵损失，减少内存使用
    """
    batch_tokens, hidden_dim = _input.shape
    vocab_size = weight.shape[0]

    # Ensure contiguous inputs
    _input = _input.contiguous()
    weight = weight.contiguous()
    target = target.contiguous()

    # Compute logits: (batch_tokens, vocab_size)
    logits = torch.matmul(_input, weight.t())
    if bias is not None:
        logits = logits + bias

    # Output tensor for losses
    losses = torch.zeros(batch_tokens, dtype=torch.float32, device=_input.device)

    # Auto-tuning block size
    BLOCK_SIZE = triton.next_power_of_2(vocab_size)
    BLOCK_SIZE = min(BLOCK_SIZE, MAX_FUSED_SIZE)

    # Grid configuration
    grid = (batch_tokens,)

    # Launch cross entropy kernel
    with torch.cuda.device(_input.device):
        _cross_entropy_forward_kernel[grid](
            logits,
            logits.stride(-2),
            target,
            target.stride(-1),
            losses,
            losses.stride(-1),
            vocab_size,
            ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Apply reduction
    if reduction == "mean":
        # Create mask for non-ignored targets
        valid_mask = target != ignore_index
        if valid_mask.sum() > 0:
            loss = losses[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=_input.device)
    elif reduction == "sum":
        valid_mask = target != ignore_index
        loss = losses[valid_mask].sum()
    else:  # "none"
        loss = losses

    return loss