import torch

def torch_kernel(input_tensor, kernel_tensor):
    """简化的PyTorch参考实现 for conv2dnchw"""
    
    # Conv2DNCHW with stride=2, padding=0
    # Input and output are in NCHW format
    output = torch.nn.functional.conv2d(
        input_tensor,
        kernel_tensor,
        stride=2,
        padding=0
    )
    return output
