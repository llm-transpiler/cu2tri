import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Triton kernel that reproduces the exact indexing and computation of the
# original CUDA average‑pooling kernel. It computes a K×K average pool on an
# NHWC tensor.
# -------------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # *float32  (input tensor, NHWC)
    pool_avg_ptr,        # *float32  (output tensor, NHWC)
    batch_size,          # int32
    channels,            # int32
    input_H,             # int32
    output_H,            # int32
    stride,              # int32
    BLOCK_SIZE: tl.constexpr,
    KSIZE: tl.constexpr,
):
    pid = tl.program_id(0)                     # block index (int64)

    # -------------------------------------------------------------
    # Thread identifiers inside the block
    # -------------------------------------------------------------
    tid = tl.arange(0, BLOCK_SIZE)             # [0, BLOCK_SIZE)  (int32)
    tid_i64 = tl.cast(tid, tl.int64)           # cast to int64 for arithmetic

    # -------------------------------------------------------------
    # Linear output offset and mask for out‑of‑range threads
    # -------------------------------------------------------------
    out_offset = pid * BLOCK_SIZE + tid_i64
    total_outputs = (
        tl.cast(batch_size, tl.int64)
        * tl.cast(output_H, tl.int64)
        * tl.cast(output_H, tl.int64)
        * tl.cast(channels, tl.int64)
    )
    mask = out_offset < total_outputs

    # -------------------------------------------------------------
    # Decode the original CUDA indexing scheme
    # -------------------------------------------------------------
    # Number of output tiles per batch (output_H is assumed divisible by 4)
    tile_per_dim = tl.cast(output_H, tl.int64) // 4          # = output_H / 4
    tiles_per_batch = tile_per_dim * tile_per_dim           # = (output_H/4)^2

    batch = pid // tiles_per_batch                          # batch index
    tile_mod = pid % tiles_per_batch                         # tile index within batch

    # thread‑level bits
    t_hi = tid_i64 >> 8            # bits 8‑9 of threadIdx.x (0..3)
    t_mid = tid_i64 >> 6           # bits 6‑9 of threadIdx.x (0..15)
    c = tid_i64 & (tl.cast(channels, tl.int64) - 1)        # channel index (lower 6 bits)

    # output spatial coordinates
    h_out = ((tile_mod * 4 + t_hi) // tile_per_dim)        # row index
    w_out = ((pid * 16 + t_mid) % tl.cast(output_H, tl.int64))  # column index

    # -------------------------------------------------------------
    # Compute base offsets for input indexing (NHWC layout)
    # -------------------------------------------------------------
    input_H_i64 = tl.cast(input_H, tl.int64)
    channels_i64 = tl.cast(channels, tl.int64)
    stride_i64 = tl.cast(stride, tl.int64)

    # offset to the beginning of the batch
    batch_offset = batch * (input_H_i64 * input_H_i64 * channels_i64)

    # offset for the top‑left corner of the pooling window
    h_offset = h_out * (input_H_i64 * stride_i64 * channels_i64)
    w_offset = w_out * (stride_i64 * channels_i64)

    base_offset = batch_offset + h_offset + w_offset + c

    # -------------------------------------------------------------
    # Accumulate sum over the K×K window
    # -------------------------------------------------------------
    sum_val = tl.cast(0.0, tl.float32)

    for rv0 in tl.static_range(KSIZE):
        for rv1 in tl.static_range(KSIZE):
            # offset of the current element inside the window
            offset = (
                base_offset
                + rv0 * (input_H_i64 * channels_i64)
                + rv1 * channels_i64
            )
            # masked load (out‑of‑range threads read 0.0)
            val = tl.load(A_ptr + offset, mask=mask, other=0.0)
            sum_val += val

    # -------------------------------------------------------------
    # Compute average and write back
    # -------------------------------------------------------------
    scale = 1.0 / (KSIZE * KSIZE)          # 1/(K*K)
    avg = sum_val * scale

    tl.store(pool_avg_ptr + out_offset, avg, mask=mask)


def triton_kernel(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int,
):
    """
    Triton entry point mirroring the original CUDA kernel signature.

    Parameters
    ----------
    input_tensor   : torch.float32 CUDA tensor, NHWC layout.
    output_tensor  : torch.float32 CUDA tensor, NHWC layout (pre‑allocated).
    batch_size     : number of images in the batch.
    channels       : number of channels (C).
    input_H        : spatial height/width of the input (square).
    kernel_size    : size of the pooling kernel (K).
    stride         : stride of the pooling operation.
    """
    # -----------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------
    assert input_tensor.is_cuda and output_tensor.is_cuda, "Tensors must be CUDA tensors"
    assert input_tensor.dtype == torch.float32 and output_tensor.dtype == torch.float32, "Only float32 supported"

    # Ensure contiguous memory layout
    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()

    # -----------------------------------------------------------------
    # Compute output spatial dimension and total number of output elements
    # -----------------------------------------------------------------
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    # -----------------------------------------------------------------
    # Launch configuration (matches __launch_bounds__(1024) in the CUDA kernel)
    # -----------------------------------------------------------------
    BLOCK_SIZE = 1024
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # -----------------------------------------------------------------
    # Launch the Triton kernel
    # -----------------------------------------------------------------
    _triton_kernel_impl[grid](
        input_tensor,
        output_tensor,
        batch_size,
        channels,
        input_H,
        output_H,
        stride,
        BLOCK_SIZE=BLOCK_SIZE,
        KSIZE=kernel_size,
        num_warps=32,   # 1024 threads per block = 32 warps
    )