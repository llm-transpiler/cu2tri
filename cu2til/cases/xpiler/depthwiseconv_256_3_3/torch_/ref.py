import torch

def torch_kernel(*args):
    """PyTorch参考实现 for depthwiseconv"""
    input_tensor, kernel_tensor = args[0], args[1]
    
    # Convert HWC to NCHW format for PyTorch
    # Input: (H, W, C) -> (1, C, H, W)
    input_nchw = input_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # Kernel: (kH, kW, C) -> (C, 1, kH, kW) for depthwise conv
    kernel_nchw = kernel_tensor.permute(2, 0, 1).unsqueeze(1)
    
    # Depthwise convolution
    output_nchw = torch.nn.functional.conv2d(
        input_nchw, 
        kernel_nchw, 
        groups=input_tensor.shape[2],  # groups = channels for depthwise
        padding=0
    )
    
    # Convert back: (1, C, H, W) -> (H, W, C)
    output = output_nchw.squeeze(0).permute(1, 2, 0)
    return output
