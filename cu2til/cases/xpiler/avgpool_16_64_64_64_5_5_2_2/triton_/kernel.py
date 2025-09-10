import torch
import triton
import triton.language as tl

@triton.jit
def avgpool2d_kernel(
    input_ptr, output_ptr,
    batch_size, channels, height, width,
    out_height, out_width,
    kernel_h, kernel_w,
    stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple 2D Average Pooling Triton kernel
    """
    idx = tl.program_id(axis=0)
    
    if idx >= batch_size * channels * out_height * out_width:
        return
    
    # Calculate output position
    temp = idx
    ow = temp % out_width
    temp = temp // out_width
    oh = temp % out_height
    temp = temp // out_height
    c = temp % channels
    b = temp // channels
    
    # Calculate input start position
    ih_start = oh * stride_h
    iw_start = ow * stride_w
    
    # Average pooling
    sum_val = 0.0
    count = 0
    
    for kh in range(kernel_h):
        ih = ih_start + kh
        if ih >= height:
            continue
        for kw in range(kernel_w):
            iw = iw_start + kw
            if iw >= width:
                continue
            
            input_idx = ((b * channels + c) * height + ih) * width + iw
            sum_val += tl.load(input_ptr + input_idx)
            count += 1
    
    output_idx = ((b * channels + c) * out_height + oh) * out_width + ow
    if count > 0:
        tl.store(output_ptr + output_idx, sum_val / count)


def triton_kernel(input_tensor, kernel_size_h, kernel_size_w, stride_h, stride_w):
    """
    Triton avgpool2d implementation
    """
    batch_size, channels, height, width = input_tensor.shape
    
    # Calculate output dimensions
    out_height = (height - kernel_size_h) // stride_h + 1
    out_width = (width - kernel_size_w) // stride_w + 1
    
    output_tensor = torch.empty((batch_size, channels, out_height, out_width), 
                               dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    total_elements = batch_size * channels * out_height * out_width
    grid = (triton.cdiv(total_elements, 1024),)
    
    avgpool2d_kernel[grid](
        input_tensor, output_tensor,
        batch_size, channels, height, width,
        out_height, out_width,
        kernel_size_h, kernel_size_w,
        stride_h, stride_w,
        BLOCK_SIZE=1024
    )
    
    return output_tensor