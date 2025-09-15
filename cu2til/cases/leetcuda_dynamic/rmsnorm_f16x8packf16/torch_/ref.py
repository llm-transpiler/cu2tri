import torch

def torch_kernel(x: torch.Tensor, g: float) -> torch.Tensor:
    # y'=x/rms(x) 1/rms(x) = rsqrtf(sum(x^2)/K)
    s_rms = torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True))
    y = (x * s_rms) * g
    return y
