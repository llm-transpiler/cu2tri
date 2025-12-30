import torch
import triton
import triton.language as tl
from packaging.version import Version

# Handle triton version compatibility for tanh
if Version(triton.__version__) >= Version("3.0.0"):
    from triton.language.extra import libdevice
    triton_tanh = libdevice.tanh
else:
    triton_tanh = tl.math.tanh

@triton.jit
def _cross_entropy_forward(
    logits_ptr,
    logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    """
    Triton accelerated cross entropy forward pass
    Based on unsloth's implementation with numerical stability
    """
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * tl.cast(logits_row_stride, tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    # Apply logit scaling for Cohere: t * x
    if DO_LOGIT_SCALING:
        logits = LOGIT_SCALE * logits

    # Apply logit softcapping for Gemma 2: t * tanh(1/t * x)
    if DO_SOFTCAPPING:
        # x_cap = t * tanh(x / t)
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    # Numerically stable logsumexp: log(sum(exp(x - max(x)))) + max(x)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx).to(tl.float32)
        # Apply same transformations to the target logit
        if DO_LOGIT_SCALING:
            x = LOGIT_SCALE * x
        if DO_SOFTCAPPING:
            x = SOFTCAP * triton_tanh(x / SOFTCAP)
        loss = logsumexp - x
    else:
        loss = 0.0

    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)


@triton.jit
def _chunked_cross_entropy_forward(
    logits_ptr,
    logits_row_stride: tl.constexpr,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    """
    Chunked cross entropy for very large vocabularies (>65K)
    Divides vocabulary into chunks to compute logsumexp efficiently
    """
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr += row_idx * tl.cast(logits_row_stride, tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx

    col_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    # Apply logit scaling for Cohere: t * x
    if DO_LOGIT_SCALING:
        logits = LOGIT_SCALE * logits

    # Apply logit softcapping for Gemma 2: t * tanh(1/t * x)
    if DO_SOFTCAPPING:
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    # Compute logsumexp for this chunk
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if chunk_idx == 0:
        # Store partial loss (will be completed after logsumexp reduction)
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            # Apply same transformations to the target logit
            if DO_LOGIT_SCALING:
                x = LOGIT_SCALE * x
            if DO_SOFTCAPPING:
                x = SOFTCAP * triton_tanh(x / SOFTCAP)
            loss = -1.0 * x  # Will add final logsumexp later
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)

    tl.store(logsumexp_ptr, logsumexp)


def triton_kernel(logits: torch.Tensor, labels: torch.Tensor, logit_softcapping: torch.Tensor, logit_scaling: torch.Tensor) -> torch.Tensor:
    """
    Triton实现：Fast Cross Entropy Loss
    基于unsloth的优化实现，支持大词汇表和数值稳定性
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device

    # Ensure contiguous inputs
    logits = logits.contiguous()
    labels = labels.contiguous()

    # Flatten for processing
    logits_flat = logits.view(batch_size * seq_len, vocab_size)
    labels_flat = labels.view(-1)

    # Extract scalar values
    softcap = logit_softcapping.item()
    scale = logit_scaling.item()

    DO_SOFTCAPPING = softcap != 0.0
    DO_LOGIT_SCALING = scale != 1.0

    MAX_FUSED_SIZE = 65536  # Maximum vocab size for single-pass processing
    div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
    n_chunks = div + (mod != 0)

    losses = torch.empty(batch_size * seq_len, dtype=torch.float32, device=device)

    if n_chunks == 1:
        # Single pass for smaller vocabularies
        BLOCK_SIZE = 4096
        num_warps = 8 if vocab_size <= 32768 else 16

        logsumexp = torch.empty(batch_size * seq_len, dtype=torch.float32, device=device)

        grid = lambda meta: (batch_size * seq_len,)
        _cross_entropy_forward[grid](
            logits_flat, logits_flat.stride(0),
            losses,
            logsumexp,
            labels_flat,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            DO_SOFTCAPPING=DO_SOFTCAPPING,
            SOFTCAP=softcap,
            DO_LOGIT_SCALING=DO_LOGIT_SCALING,
            LOGIT_SCALE=scale,
            num_warps=num_warps,
        )
    else:
        # Multi-pass for very large vocabularies
        BLOCK_SIZE = MAX_FUSED_SIZE
        logsumexp = torch.empty((batch_size * seq_len, n_chunks), dtype=torch.float32, device=device)

        grid = lambda meta: (batch_size * seq_len, n_chunks)
        _chunked_cross_entropy_forward[grid](
            logits_flat, logits_flat.stride(0),
            losses,
            logsumexp,
            labels_flat,
            VOCAB_SIZE=vocab_size,
            N_CHUNKS=n_chunks,
            BLOCK_SIZE=BLOCK_SIZE,
            DO_SOFTCAPPING=DO_SOFTCAPPING,
            SOFTCAP=softcap,
            DO_LOGIT_SCALING=DO_LOGIT_SCALING,
            LOGIT_SCALE=scale,
            num_warps=16,
        )

        # Final logsumexp reduction across chunks
        logsumexp = torch.logsumexp(logsumexp, dim=1)
        losses += logsumexp
        losses.masked_fill_(labels_flat == -100, 0)

    # Compute mean over non-padding tokens
    n_items = (labels_flat != -100).sum()
    if n_items > 0:
        return losses.sum() / n_items
    else:
        return torch.tensor(0.0, device=device, dtype=torch.float32)