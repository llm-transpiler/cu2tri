import torch

def torch_kernel(input_tensor, kernel_tensor, stride=2, padding=0):
    output = torch.nn.functional.conv2d(
        input_tensor,
        kernel_tensor,
        stride=stride,
        padding=padding
    )
    return output
