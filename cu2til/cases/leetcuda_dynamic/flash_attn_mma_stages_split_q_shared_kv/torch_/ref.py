import torch
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func

def flash_attn_kernel(Q, K, V):
    return flash_attn_func(Q, K, V, causal=False, dropout_p=0.0)