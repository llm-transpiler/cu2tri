import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(input_ptr, kernel_ptr, output_ptr, 
                       batch_size, input_height, input_channels, 
                       output_channels, kernel_height, stride,
                       BLOCK_SIZE_OC: tl.constexpr):
    # Get thread indices
    oc = tl.thread_id(0)  # output channel index
    oh = tl.block_id(1)   # output height index
    ow = tl.block_id(2)   # output width index
    bs = tl.block_id(3)   # batch index
    
    # Check bounds
    if oc >= output_channels or oh >= (input_height - kernel_height) // stride + 1 or \
       ow >= (input_height - kernel_height) // stride + 1 or bs >= batch_size:
        return
    
    # Calculate output dimensions
    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_height - kernel_height) // stride + 1
    
    # Initialize accumulator
    sum = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel spatial dimensions and input channels
    for kh in range(kernel_height):
        for kw in range(kernel_height):
            for ic in range(input_channels):
                # Calculate input indices
                ih = oh * stride + kh
                iw = ow * stride + kw
                
                # Calculate input index
                input_idx = bs * (input_height * input_height * input_channels) + \
                           ih * (input_height * input_channels) + \
                           iw * input_channels + ic
                
                # Calculate kernel index
                kernel_idx = oc * (kernel_height * kernel_height * input_channels) + \
                            kh * (kernel_height * input_channels) + \
                            kw * input_channels + ic
                
                # Load values and accumulate
                input_val = tl.load(input_ptr + input_idx)
                kernel_val = tl.load(kernel_ptr + kernel_idx)
                sum = sum + input_val * kernel_val
    
    # Calculate output index
    output_idx = bs * (output_height * output_width * output_channels) + \
                oh * (output_width * output_channels) + \
                ow * output_channels + oc
    
    # Store result
    tl.store(output_ptr + output_idx, sum[0])

def triton_kernel(input, filter, output, 
                  batch_size, input_height,
                  input_channels, output_channels,
                  kernel_height, stride):
    # Calculate output dimensions
    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_height - kernel_height) // stride + 1
    
    # Define block size for output channels
    BLOCK_SIZE_OC = output_channels
    
    # Configure grid dimensions
    grid = (
        triton.cdiv(output_channels, BLOCK_SIZE_OC),  # number of blocks for OC
        output_height,  # output height
        output_width,   # output width
        batch_size      # batch size
    )
    
    # Launch kernel
    _triton_kernel_impl[grid](
        input, filter, output,
        batch_size, input_height, input_channels,
        output_channels, kernel_height, stride,
        BLOCK_SIZE_OC=BLOCK_SIZE_OC
    )