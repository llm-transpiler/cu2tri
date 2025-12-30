import torch
import triton
import triton.language as tl

@triton.jit
def _fg_kernel(e, g, h, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    SwiGLU forward kernel
    SwiGLU(gate, up) = Swish(gate) * up where Swish(x) = x * sigmoid(x)
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load gate and up projections
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # Compute Swish activation: f = e * sigmoid(e)
    f_row = e_row * tl.sigmoid(e_row)
    f_row = f_row.to(g_row.dtype)  # Match input precision

    # SwiGLU: h = f * g
    h_row = f_row * g_row

    # Store result
    tl.store(h + offsets, h_row, mask=mask)


def triton_kernel(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Triton实现：SwiGLU (Swish-Gated Linear Unit)
    基于unsloth的实现，SwiGLU(gate, up) = Swish(gate) * up
    """
    batch_size, seq_len, hidden_dim = gate.shape
    n_elements = gate.numel()
    device = gate.device

    # Ensure contiguous inputs
    gate = gate.contiguous()
    up = up.contiguous()

    # Output tensor
    h = torch.empty((batch_size, seq_len, hidden_dim), dtype=gate.dtype, device=device)

    # Grid configuration
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    with torch.cuda.device(device):
        _fg_kernel[grid](
            gate, up, h, n_elements,
            BLOCK_SIZE=1024,
        )

    return h