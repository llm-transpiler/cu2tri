import torch
import triton
import triton.language as tl


@triton.jit
def silu(x):
    """SiLU activation function: x * sigmoid(x)"""
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Liger-Kernel SwiGLU forward kernel
    SwiGLU(gate, up) = SiLU(gate) * up
    """
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row).cast(b_row.dtype) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


def triton_kernel(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Triton实现：SwiGLU (基于Liger-Kernel)
    高性能Swish-Gated Linear Unit实现
    """
    batch_size, seq_len, hidden_dim = gate.shape
    n_elements = gate.numel()
    device = gate.device

    # Ensure contiguous inputs
    gate = gate.contiguous()
    up = up.contiguous()

    # Output tensor
    out = torch.empty((batch_size, seq_len, hidden_dim), dtype=gate.dtype, device=device)

    # Auto-tuning block size
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)  # Cap at reasonable size

    # Grid configuration
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    with torch.cuda.device(device):
        _swiglu_forward_kernel[grid](
            gate,
            up,
            out,
            gate.stride(-1),
            hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out