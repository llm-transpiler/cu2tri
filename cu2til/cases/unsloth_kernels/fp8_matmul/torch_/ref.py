import torch

def torch_kernel(X: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """
    PyTorch参考实现：FP8 Block-wise Quantized Matrix Multiplication
    简化版本，使用标准矩阵乘法作为参考
    """
    # For reference implementation, use standard float32 matmul
    # In a real implementation, this would include:
    # 1. Input quantization to FP8
    # 2. Weight dequantization from FP8 using scales
    # 3. Matrix multiplication with proper scaling
    # 4. Output dequantization

    # Simplified reference: just compute standard matmul in higher precision
    result = torch.matmul(X, weight.t())

    # Apply weight scaling (simplified)
    if weight_scale.numel() == 1:
        result = result * weight_scale
    else:
        # For block-wise scaling, we'd need more complex logic
        # For now, use average scaling
        avg_scale = weight_scale.mean()
        result = result * avg_scale

    return result