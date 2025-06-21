
"""
结果分析模块

负责比较和分析内核执行结果的正确性
"""

from typing import Dict, Any

def calculate_performance_metrics(
    triton_time: float,
    cuda_time: float,
    torch_time: float
) -> Dict[str, float]:
    """
    计算性能指标
    
    Args:
        triton_time: Triton内核执行时间（毫秒）
        cuda_time: CUDA内核执行时间（毫秒）
        torch_time: PyTorch参考执行时间（毫秒）
        
    Returns:
        性能指标字典
    """
    performance_metrics = {
        "triton_time": triton_time,
        "cuda_time": cuda_time,
        "torch_time": torch_time
    }
    
    if triton_time > 0:
        performance_metrics.update({
            "triton_cuda_speedup": cuda_time / triton_time,
            "triton_torch_speedup": torch_time / triton_time
        })
    
    return performance_metrics


def _check_numerical_match(
    val1: float, 
    val2: float, 
    atol: float, 
    rtol: float
) -> bool:
    diff = abs(val1 - val2)
    return diff <= atol + rtol * abs(val2)


def format_correctness_report(correctness: Dict[str, Any]) -> str:
    """
    格式化正确性报告
    
    Args:
        correctness: 正确性检查结果
        
    Returns:
        格式化的报告字符串
    """
    def status_icon(match: bool) -> str:
        return "✅" if match else "❌"
    
    lines = [
        f"🔍 Triton vs PyTorch: {status_icon(correctness.get('triton_torch_match', False))}",
        f"🔍 Triton vs CUDA: {status_icon(correctness.get('triton_cuda_match', False))}"
    ]
    
    # 添加sum比较信息（用于调试）
    if 'triton_sum' in correctness:
        lines.append(f"🔢 Triton sum: {correctness['triton_sum']:.6f}")
    if 'cuda_sum' in correctness:
        lines.append(f"🔢 CUDA sum: {correctness['cuda_sum']:.6f}")
    if 'torch_sum' in correctness:
        lines.append(f"🔢 PyTorch sum: {correctness['torch_sum']:.6f}")
    
    return "\n".join(lines)


def format_performance_report(performance: Dict[str, float]) -> str:
    """
    格式化性能报告
    
    Args:
        performance: 性能测试结果
        
    Returns:
        格式化的报告字符串
    """
    lines = [
        "📈 Performance Test Results:",
        f"   Triton:   {performance['triton_time']:.3f} ms",
        f"   CUDA:     {performance['cuda_time']:.3f} ms",
        f"   PyTorch:  {performance['torch_time']:.3f} ms"
    ]
    
    if 'triton_cuda_speedup' in performance:
        lines.extend([
            "\n🚀 Speedup:",
            f"   Triton vs CUDA: {performance['triton_cuda_speedup']:.2f}x",
            f"   Triton vs PyTorch: {performance['triton_torch_speedup']:.2f}x"
        ])
    
    return "\n".join(lines) 