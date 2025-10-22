import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr,
    output_ptr,
    inner,
    rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < inner

    # Initialize with +inf (equivalent to FLT_MAX)
    INF = float('inf')
    min_val = tl.full([BLOCK_SIZE], INF, dtype=tl.float32)

    # Compile‑time unrolled reduction over rows
    for r in range(rows):
        idx = r * inner + offsets
        x = tl.load(input_ptr + idx, mask=mask, other=min_val)
        min_val = tl.minimum(min_val, x)

    # Write the per‑column minima
    tl(output_ptr + offsets, min_val, mask=mask)

def triton_kernel(input, output, rows, inner):
    """
    Triton wrapper that computes the per‑column minimum across `rows` rows.
    Mirrors the signature of the original CUDA kernel:
        void cuda_kernel(const float* input, float* output, int rows, int inner)

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape (rows, inner) on CUDA device, dtype torch.float32.
    output : torch.Tensor
        Output tensor of shape (inner,) on CUDA device, dtype torch.float32.
    rows : int
        Number of rows in the input tensor.
    inner : int
        Number of columns (inner dimension) in the input tensor.
    """
    if not input.is_cuda or not output.is_cuda:
        raise RuntimeError("Input and output tensors must be on CUDA device")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("Input and output tensors must be torch.float32")
    if input.shape != (rows, inner):
        raise RuntimeError(f"Input shape {input.shape} does not match (rows={rows}, inner={inner})")
    if output.shape != (inner,):
        raise RuntimeError(f"Output shape {output.shape} does not match (inner={inner})")
    if not input.is_contiguous():
        input = input.contiguous()
    if not output.is_contiguous():
        output = output.contiguous()

    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(inner, meta['BLOCK_SIZE']),)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        output,
        inner,
        rows=rows,
        BLOCK_SIZE=BLOCK_SIZE
    )
    torch.cuda.synchronize()

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    rows = 1024
    inner = 4096

    # Random input tensor on GPU
    input_tensor = torch.rand(rows, inner, dtype=torch.float32, device='cuda')
    # Allocate output tensor on GPU
    output_tensor = torch.empty(inner, dtype=torch.float32, device='cuda')

    # Expected result using PyTorch
    expected = torch.min(input_tensor, dim=0).values

    # Run Triton kernel
    triton_kernel(input_tensor, output_tensor, rows, inner)

    # Verify correctness
    torch.testing.assert_allclose(output_tensor, expected, atol=1e-6, rtol=1e-6)
    print("Test passed!")