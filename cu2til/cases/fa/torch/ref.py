import torch
import torch.nn.functional as F
import math

def naive_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y