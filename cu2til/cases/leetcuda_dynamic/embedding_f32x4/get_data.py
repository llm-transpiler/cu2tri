import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Embedding f32x4 parameters for dynamic shapes"""
    test_shapes: List[Tuple[int, int, int]] = None  # List of (n, vocab_size, emb_size) shapes to test
    
    def __post_init__(self):
        if self.test_shapes is None:
            # Default test shapes covering various (n, vocab_size, emb_size) combinations
            self.test_shapes = [
                (128, 512, 256),   # Small
                (256, 1024, 512),  # Medium
                (512, 1024, 512),  # Medium seq
                (256, 2048, 512),  # Large vocab
                (256, 1024, 1024), # Large emb
                (512, 2048, 512),  # Large seq+vocab
                (1024, 1024, 512), # Very large seq
                (256, 4096, 1024), # Very large all
            ]

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # idx (GPU pointer)
        ctypes.c_void_p,  # weight (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,     # n (sequence length)
        ctypes.c_int      # emb_size (embedding dimension)
    ]

def get_cuda_torch_inputs_for_config(config: Tuple[int, int, int]):
    """Create input data for both CUDA and PyTorch implementations for a specific config"""
    torch.manual_seed(SEED)
    
    n, vocab_size, emb_size = config
    
    # Create input indices (token indices)
    idx = torch.randint(0, vocab_size, (n,), dtype=torch.int32, device="cuda")
    
    # Create embedding weight matrix
    weight = torch.randn((vocab_size, emb_size), dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.1)
    
    # Create output tensor
    output_cuda = torch.empty((n, emb_size), dtype=torch.float32, device="cuda")
    
    # CUDA inputs: GPU pointers list
    cuda_all_inputs = [
        idx,
        weight,
        output_cuda,
        n,
        emb_size
    ]
    
    # PyTorch inputs: indices and weight matrix for reference implementation
    torch_all_inputs = [idx, weight]
    
    # CUDA output tensor for result comparison
    cuda_output_tensors = [output_cuda]
    
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def get_all_cuda_torch_inputs(params: Params):
    """Create input data for all test shapes"""
    all_test_data = []
    
    for config in params.test_shapes:
        test_data = get_cuda_torch_inputs_for_config(config)
        # Use config as shape info for consistency with the framework
        all_test_data.append((config, test_data))
    
    return all_test_data

def cuda_output_tensor_transform(cuda_output):
    return cuda_output
