import torch
import numpy as np

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / np.pi)) * (x + 0.044715 * torch.pow(x, 3))))
