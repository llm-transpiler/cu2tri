import torch
import triton
import triton.language as tl

@triton.jit
def sumpool_kernel(input_ptr, output_ptr, batch_size, channels, input_h, kernel_size, stride, 
                   BLOCK_SIZE: tl.constexpr):
    # Get the current program id
    pid = tl.program_id(axis=0)
    
    # Calculate output dimensions
    output_h = (input_h - kernel_size) // stride + 1
    output_w = output_h  # assuming square input/output
    
    # Calculate total output elements per sample
    output_elements_per_batch = output_h * output_w * channels
    total_output_elements = batch_size * output_elements_per_batch
    
    # Compute the block start offset
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for elements within bounds
    mask = offsets < total_output_elements
    
    # For each output element, we need to compute the sum of the corresponding window
    for i in range(BLOCK_SIZE):
        if block_start + i >= total_output_elements:
            break
            
        # Convert linear index to 4D coordinates (batch, h, w, c)
        idx = block_start + i
        batch_idx = idx // output_elements_per_batch
        remaining = idx % output_elements_per_batch
        
        c_idx = remaining % channels
        remaining = remaining // channels
        w_idx = remaining % output_w
        h_idx = remaining // output_w
        
        # Calculate the window sum
        window_sum = 0.0
        
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                in_h = h_idx * stride + kh
                in_w = w_idx * stride + kw
                
                if in_h < input_h and in_w < input_h:  # assuming square input
                    # NHWC format: (batch, height, width, channels)
                    input_idx = (batch_idx * input_h * input_h * channels + 
                                in_h * input_h * channels + 
                                in_w * channels + 
                                c_idx)
                    
                    input_val = tl.load(input_ptr + input_idx)
                    window_sum += input_val
        
        # Store the sum value
        tl.store(output_ptr + idx, window_sum)

def triton_kernel(input_nhwc, output_nhwc, batch_size, channels, input_h, kernel_size, stride):
    # Calculate total output elements
    output_h = (input_h - kernel_size) // stride + 1
    output_w = output_h
    total_output_elements = batch_size * output_h * output_w * channels
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(total_output_elements, BLOCK_SIZE),)
    
    # Launch kernel
    sumpool_kernel[grid](
        input_nhwc, output_nhwc, batch_size, channels, input_h, kernel_size, stride,
        BLOCK_SIZE=BLOCK_SIZE
    )
