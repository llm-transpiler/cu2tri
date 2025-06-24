"""
基准测试模块

提供内核性能测试功能
"""

from typing import List, Callable

import torch, triton

def benchmark_kernel(
    kernel_func: Callable, 
    inputs: list | tuple,
    warmup: int = 1000,
    iterations: int = 10000,
    quantiles: List[float] = [0.2, 0.5, 0.8]
) -> float:
    """
    测试内核性能
    
    Args:
        kernel_func: 要测试的内核函数
        inputs: 输入张量列表
        config: 测试配置
        
    Returns:
        平均执行时间（毫秒）
    """
    return _simple_perf(kernel_func, inputs, warmup, iterations, quantiles, get_mean=True)


def _simple_perf(
    kernel_func: Callable, 
    inputs: list | tuple,
    warmup: int = 1000,
    iterations: int = 10000,
    quantiles: List[float] = [0.2, 0.5, 0.8],
    get_mean: bool = True
) -> float | dict:
    # 预热
    for _ in range(warmup):
        kernel_func(*inputs)
    
    torch.cuda.synchronize()
    
    # 创建events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    # 测量时间
    torch.cuda.synchronize()
    for i in range(iterations):
        cache.zero_()
        start_events[i].record()
        kernel_func(*inputs)
        end_events[i].record()
    torch.cuda.synchronize()

    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_events, end_events)], dtype=torch.float)
    
    if get_mean:
        return torch.mean(times).item()

    ret_quantiles = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
    
    return {
        'mean': torch.mean(times).item(),
        'total': torch.sum(times).item(),
        'var': torch.var(times).item(),
        'median': torch.median(times).item(),
        **{f'q{int(q*100)}': ret_quantiles[i] for i, q in enumerate(quantiles)}
    }