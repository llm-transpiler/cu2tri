"""
基准测试模块

提供内核性能测试功能
"""

from typing import List, Callable

import torch, triton

def benchmark_kernel(
    kernel_func: Callable, 
    inputs: list | tuple,
    warmup: int = 100,
    iterations: int = 500,
    quantiles: List[float] = [0.2, 0.5, 0.8]
) -> float:
    ret = []
    for _ in range(10):
        res = _simple_perf(kernel_func, inputs, warmup, iterations, quantiles)
        ret.append(res["median"])
    ret = torch.tensor(ret, dtype=torch.float)
    return ret.median().item()

def benchmark_kernel_cudagraph(
    kernel_func: Callable, 
    inputs: list | tuple,
    warmup: int = 1000,
    iterations: int = 1000,
    n_retries: int = 100,
    quantiles: List[float] = [0.2, 0.5, 0.8]
) -> float:
    res = _simple_perf_cudagraph(kernel_func, inputs, warmup, iterations, n_retries, quantiles)
    return res["median"]

def benchmark_simple_e2e(
    kernel_func: Callable, 
    inputs: list | tuple,
    warmup: int = 10,
    iterations: int = 50,
) -> float:
    for _ in range(warmup):
        kernel_func(*inputs)
    
    torch.cuda.synchronize()
    
    # 创建events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # torch.cuda.synchronize()
    start_event.record()
    for _ in range(iterations):
        kernel_func(*inputs)
    end_event.record()
    torch.cuda.synchronize()
    
    # for _ in range(5):
    #     start_event.record()
    #     kernel_func(*inputs)
    #     end_event.record()
    #     torch.cuda.synchronize()
    #     time += start_event.elapsed_time(end_event)
    # time = time / 5

    time = start_event.elapsed_time(end_event)
    res = time / iterations
    return res
    
def _simple_perf(
    kernel_func: Callable, 
    inputs: list | tuple,
    warmup: int = 1000,
    iterations: int = 1000,
    quantiles: List[float] = [0.2, 0.5, 0.8],
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
    
    # if get_mean:
    #     return torch.mean(times).item()

    ret_quantiles = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
    
    res = {
        'mean': torch.mean(times).item(),
        'total': torch.sum(times).item(),
        'var': torch.var(times).item(),
        'median': torch.median(times).item(),
        **{f'q{int(q*100)}': ret_quantiles[i] for i, q in enumerate(quantiles)}
    }
    
    return res


def _simple_perf_cudagraph(
    kernel_func: Callable, 
    inputs: list | tuple,
    warmup: int = 1000,
    iterations: int = 1000,
    n_retries: int = 100,
    quantiles: List[float] = [0.2, 0.5, 0.8],
) -> float | dict:
    # triton.testing.do_bench_cudagraph
    with torch.cuda.stream(torch.cuda.Stream()):
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                inp.detach_()
                inp.requires_grad_(True)
                inp.grad = None
        for _ in range(warmup):
            kernel_func(*inputs)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iterations):
                kernel_func(*inputs)
        torch.cuda.synchronize()
        # measure time and return
        ret = []
        for _ in range(n_retries):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            g.replay()
            end_event.record()
            torch.cuda.synchronize()
            ret += [start_event.elapsed_time(end_event) / iterations]

    times = torch.tensor(ret, dtype=torch.float)
    ret_quantiles = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
    res = {
        'mean': torch.mean(times).item(),
        'total': torch.sum(times).item(),
        'var': torch.var(times).item(),
        'median': torch.median(times).item(),
        **{f'q{int(q*100)}': ret_quantiles[i] for i, q in enumerate(quantiles)}
    }

    return res