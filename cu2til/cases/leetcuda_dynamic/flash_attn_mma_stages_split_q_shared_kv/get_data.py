import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED
import math

@dataclass
class Params:
    """Flash Attention MMA stages split Q shared KV parameters for dynamic shapes"""
    test_shapes: List[Tuple[int, int, int, int]] = None  # List of (batch_size, num_heads, seq_len, head_dim) shapes to test
    
    def __post_init__(self):
        if self.test_shapes is None:
            # Generate comprehensive test shapes based on test_fa.py and test_kernel.py
            self.test_shapes = self._generate_test_shapes()
    
    def _generate_test_shapes(self):
        """Generate test shapes based on configurations from test files"""
        batch_size_list = [1, 4, 8]
        head_num_list = [1, 4, 8]  
        seq_len_list = [1024, 2048, 4096]
        head_dim_list = [64, 128]
        
        # Basic combinations from test files
        test_shapes = []
        for B in batch_size_list:
            for H in head_num_list:
                for N in seq_len_list:
                    for D in head_dim_list:
                        test_shapes.append((B, H, N, D))
        
        # Additional shapes for comprehensive testing (from get_test_shapes in test files)
        dim = 2048
        bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
        
        for headdim in [64, 128, 256]:
            if headdim > 128:  # Skip 256 for now as it's not supported in current kernel
                continue
            nheads = dim // headdim
            for batch_size, seqlen in bs_seqlen_vals:
                test_shapes.append((batch_size, nheads, seqlen, headdim))
        
        # Additional small test shapes for quick validation
        quick_test_shapes = [
            (1, 4, 128, 32),   # Small
            (2, 8, 256, 64),   # Medium
            (1, 8, 512, 64),   # Medium seq
            (4, 8, 256, 64),   # Large batch
            (2, 16, 512, 64),  # Many heads
            (1, 8, 1024, 64),  # Long seq
            (4, 16, 512, 64),  # Large batch+heads
            (2, 8, 2048, 128), # Very large
        ]
        
        # Combine all shapes and remove duplicates
        all_shapes = test_shapes + quick_test_shapes
        return list(set(all_shapes))

def get_cuda_argtypes():
    """Return ctypes argument types for the CUDA kernel function
    
    Function signature:
    extern "C" void flash_attn_mma_stages_split_q_shared_kv(half* Q, half* K, half* V, half* O,
                                                            int batch_size, int num_heads, 
                                                            int seq_len, int head_dim)
    """
    return [
        ctypes.c_void_p,  # Q (GPU pointer)
        ctypes.c_void_p,  # K (GPU pointer)
        ctypes.c_void_p,  # V (GPU pointer)
        ctypes.c_void_p,  # O (GPU pointer)
        ctypes.c_int,     # batch_size
        ctypes.c_int,     # num_heads
        ctypes.c_int,     # seq_len
        ctypes.c_int      # head_dim
    ]

def get_cuda_torch_inputs_for_config(config: Tuple[int, int, int, int]):
    """Create input data for both CUDA and PyTorch implementations for a specific config
    
    Returns:
        cuda_all_inputs: List containing [Q_tensor, K_tensor, V_tensor, O_tensor, batch_size, num_heads, seq_len, head_dim]
        torch_all_inputs: List containing [Q, K, V] for reference PyTorch implementation
        cuda_output_tensors: List containing [O_cuda] for result comparison
    """
    torch.manual_seed(SEED)
    
    batch_size, num_heads, seq_len, head_dim = config
    qkv_shape = (batch_size, num_heads, seq_len, head_dim)
    output_shape = (batch_size, num_heads, seq_len, head_dim)
    
    # Create Q, K, V tensors with half precision
    # Shape: [batch_size, num_heads, seq_len, head_dim]
    # Using smaller values for better numerical stability, similar to test files
    Q = torch.randn(qkv_shape, dtype=torch.float16, device="cuda") * 0.1
    K = torch.randn(qkv_shape, dtype=torch.float16, device="cuda") * 0.1  
    V = torch.randn(qkv_shape, dtype=torch.float16, device="cuda") * 0.1
    
    # Create output tensor (initialized to zeros like in test files)
    O_cuda = torch.zeros(output_shape, dtype=torch.float16, device="cuda")
    
    # CUDA inputs: tensor objects (will be converted to pointers when calling kernel) + parameters
    cuda_all_inputs = [
        Q,              # Will be converted to Q.data_ptr()
        K,              # Will be converted to K.data_ptr() 
        V,              # Will be converted to V.data_ptr()
        O_cuda,         # Will be converted to O_cuda.data_ptr()
        batch_size,     # int
        num_heads,      # int
        seq_len,        # int
        head_dim        # int
    ]
    
    # PyTorch inputs: tensors for reference implementation (naive_attn from test files)
    torch_all_inputs = [Q, K, V]
    
    # CUDA output tensor for result comparison
    cuda_output_tensors = [O_cuda]
    
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def get_all_cuda_torch_inputs(params: Params):
    """Create input data for all test shapes"""
    all_test_data = []
    
    for config in params.test_shapes:
        test_data = get_cuda_torch_inputs_for_config(config)
        all_test_data.append((config, test_data))
    
    return all_test_data

def get_qkv(batch_size: int, num_heads: int, seq_len: int, head_dim: int):
    """Get Flash Attention QKV matrices (similar to test_fa.py and test_kernel.py)
    
    Args:
        batch_size: Batch size B
        num_heads: Number of attention heads H  
        seq_len: Sequence length N
        head_dim: Head dimension D
        
    Returns:
        Tuple of (Q, K, V) tensors with shape [B, H, N, D]
    """
    torch.manual_seed(SEED)
    q = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=torch.half, device="cuda")
    k = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=torch.half, device="cuda")
    v = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=torch.half, device="cuda")
    return q, k, v

def get_comprehensive_test_shapes():
    """Get comprehensive test shapes covering various scenarios (based on test files)"""
    batch_size_list = [1, 4, 8]
    head_num_list = [1, 4, 8]
    seq_len_list = [1024, 2048, 4096]
    head_dim_list = [64, 128]
    
    test_shapes = []
    for B in batch_size_list:
        for H in head_num_list:
            for N in seq_len_list:
                for D in head_dim_list:
                    test_shapes.append((B, H, N, D))
    
    # Additional comprehensive shapes
    dim = 2048
    bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
    
    for headdim in [64, 128]:  # Skip 256 as it's not supported yet
        nheads = dim // headdim
        for batch_size, seqlen in bs_seqlen_vals:
            test_shapes.append((batch_size, nheads, seqlen, headdim))
    
    return list(set(test_shapes))  # Remove duplicates

def cuda_output_tensor_transform(cuda_output):
    """Transform CUDA output tensor (identity function for now)"""
    return cuda_output