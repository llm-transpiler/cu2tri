import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing the concat operation (identical to the CUDA kernel)
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    input1_ptr,          # const float* __restrict__ input1
    input2_ptr,          # const float* __restrict__ input2
    output_ptr,          # float* __restrict__ output
    N, C, H, W,          # int N, C, H, W
    total,               # int total = N * (C*2) * H * W
    BLOCK_SIZE: tl.constexpr  # compile‑time block size
):
    # 1‑D grid of blocks
    pid = tl.program_id(0)

    # Linear indices handled by this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    # Pre‑computed strides
    hw = H * W                # spatial size
    cout = C * 2              # output channel count

    # Decompose linear index into (n, c, offset)
    n = offsets // (cout * hw)
    rem = offsets % (cout * hw)
    c = rem // hw
    offset = rem % hw

    # Base offset for the corresponding input tensors (both inputs share N and spatial layout)
    base_input = n * C * hw

    # Determine which input to read from
    c_is_input1 = c < C          # True -> read from input1, False -> read from input2
    c_is_input2 = c >= C

    # Safe channel indices for each input (avoid negative offsets)
    c1 = tl.where(c_is_input1, c, 0)          # channel index for input1 (0 when reading from input2)
    c2 = tl.where(c_is_input2, c - C, 0)      # channel index for input2 (0 when reading from input1)

    # Compute safe offsets into the two inputs
    input1_offset = base_input + c1 * hw + offset
    input2_offset = base_input + c2 * hw + offset

    # Load values with appropriate masks
    val1 = tl.load(input1_ptr + input1_offset,
                   mask=mask & c_is_input1,
                   other=0.0)
    val2 = tl.load(input2_ptr + input2_offset,
                   mask=mask & c_is_input2,
                   other=0.0)

    # Select the correct value
    val = tl.where(c_is_input1, val1, val2)

    # Write back to output
    tl.store(output_ptr + offsets, val, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(
    input1: torch.Tensor,
    input2: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    H: int,
    W: int
):
    """
    Triton implementation of the concat_kernel.
    Arguments:
        input1 (torch.Tensor): shape (N, C, H, W), dtype=torch.float32, CUDA tensor
        input2 (torch.Tensor): shape (N, C, H, W), dtype=torch.float32, CUDA tensor
        output (torch.Tensor): shape (N, 2*C, H, W), dtype=torch.float32, CUDA tensor
        N, C, H, W (int): dimensions of the input tensors
    """
    # Basic sanity checks
    assert input1.is_cuda and input2.is_cuda and output.is_cuda, "All tensors must be on CUDA"
    assert input1.dtype == torch.float32 and input2.dtype == torch.float32 and output.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert input1.shape == (N, C, H, W), f"input1 shape mismatch: expected {(N, C, H, W)}, got {input1.shape}"
    assert input2.shape == (N, C, H, W), f"input2 shape mismatch: expected {(N, C, H, W)}, got {input2.shape}"
    assert output.shape == (N, C * 2, H, W), f"output shape mismatch: expected {(N, C*2, H, W)}, got {output.shape}"

    total = N * (C * 2) * H * W
    BLOCK_SIZE = 256
    grid = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input1,            # input1_ptr
        input2,            # input2_ptr
        output,            # output_ptr
        N, C, H, W,       # dimensions
        total,             # total number of output elements
        BLOCK_SIZE=BLOCK_SIZE
    )