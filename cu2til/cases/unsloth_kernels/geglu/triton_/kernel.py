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
def _exact_forward_kernel(e, g, h, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Exact GEGLU forward kernel
    GEGLU(gate, up) = Swish(gate) * up where Swish(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load gate and up projections
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # Compute Swish activation: f = 0.5 * e * (1 + erf(1/sqrt(2) * e))
    f_row = 0.5 * e_row * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = f_row.to(g_row.dtype)  # Match input precision

    # GEGLU: h = f * up
    h_row = f_row * g_row

    # Store result
    tl.store(h + offsets, h_row, mask=mask)


@triton.jit
def _approx_forward_kernel(e, g, h, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Approximate GEGLU forward kernel
    Uses tanh approximation: Swish(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * x * (1 + 0.044715 * x²)))
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load gate and up projections
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # Constants for tanh approximation
    s = 0.7978845608028654  # sqrt(2/π)

    # Approximate Swish: f = 0.5 * e * (1 + tanh(sqrt(2/π) * e * (1 + 0.044715 * e²)))
    f_row = 0.5 * e_row * (
        triton_tanh(s * e_row * (1.0 + 0.044715 * e_row * e_row)) + 1.0
    )
    f_row = f_row.to(g_row.dtype)  # Match input precision

    # GEGLU: h = f * up
    h_row = f_row * g_row

    # Store result
    tl.store(h + offsets, h_row, mask=mask)


def triton_kernel(gate: torch.Tensor, up: torch.Tensor, approximate: bool = False) -> torch.Tensor:
    """
    Triton实现：GEGLU (Gated Exponential Linear Unit)
    基于unsloth的实现，支持exact和approximate两种模式
    """
    batch_size, seq_len, hidden_dim = gate.shape
    n_elements = gate.numel()
    device = gate.device

    # Ensure contiguous inputs
    gate = gate.contiguous()
    up = up.contiguous()

    # Output tensor
    out = torch.empty((batch_size, seq_len, hidden_dim), dtype=gate.dtype, device=device)

    # Grid configuration
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Choose kernel based on approximation mode
    if approximate:
        with torch.cuda.device(device):
            _approx_forward_kernel[grid](
                gate, up, out, n_elements,
                BLOCK_SIZE=1024,
            )
    else:
        with torch.cuda.device(device):
            _exact_forward_kernel[grid](
                gate, up, out, n_elements,
                BLOCK_SIZE=1024,
            )

    return out