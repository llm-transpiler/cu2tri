import torch
import triton
import triton.language as tl
import operator

# Handle triton version compatibility for tanh
if triton.__version__ >= "3.0.0":
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh


@triton.jit
def _geglu_tanh_forward_kernel(a, b, c, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Liger-Kernel GEGLU forward kernel (tanh approximation)
    GEGLU(gate, up) = Swish(gate) * up where Swish(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a += program_id * stride
    b += program_id * stride
    c += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    # tanh approximation form of GELU is computed with:
    # 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    c_row = geglu_a.cast(b_row.dtype) * b_row
    tl.store(c + col_offsets, c_row, mask=mask)


def triton_kernel(gate: torch.Tensor, up: torch.Tensor, approximate: bool = True) -> torch.Tensor:
    """
    Triton实现：GEGLU (基于Liger-Kernel)
    基于Liger-Kernel的高性能实现，使用tanh近似
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
        _geglu_tanh_forward_kernel[grid](
            gate,
            up,
            out,
            gate.stride(-1),
            hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out