import torch
import triton

from abc import ABC, abstractmethod
from typing import Tuple
import logging
import inspect
from functools import partial

class KernelPerformer(ABC):
    @abstractmethod
    def gbps_fn(self):
        pass

    """Comprehensive kernel testing"""
    @classmethod
    def test_performance(cls, kernel_func: callable, args: tuple, kernel_name: str,
                        warmup: int, iterations: int, logger: logging.Logger = None, **kwargs) -> Tuple[float, float]:
        """Test performance and compute bandwidth"""
        try:
            gbps_fn = cls.gbps_fn()
            mode = kwargs.get('mode', 'mean')
            lambda_params = inspect.signature(gbps_fn).parameters
                        
            # 筛选出 kwargs 中那些 lambda 实际拥有的参数
            valid_args_for_lambda = {}
            for p_name in lambda_params:
                if p_name in kwargs:
                    valid_args_for_lambda[p_name] = kwargs[p_name]

            # 如果找到了任何有效参数，则创建偏函数
            if valid_args_for_lambda:
                gbps_fn = partial(gbps_fn, **valid_args_for_lambda)
            else:
                gbps_fn = gbps_fn # 如果 kwargs 中没有 lambda 的参数，则使用原始 lambda

            result = KernelPerfBench.func_perf_test(kernel_func, args, warmup, iterations)
            exec_time = result[mode]
            return exec_time, gbps_fn(exec_time)
            
        except Exception as e:
            import traceback
            if logger:
                logger.error(f"Error details: {traceback.format_exc()}")
            if logger:
                logger.error(f"{kernel_name} performance test failed: {e}")
            raise e

class KernelPerfBench:
    '''
    逐次同步方差较大
    缓存清理大幅增加测量时间
    method1 批量执行+单词测量时间偏短
    
    '''
    di = torch.cuda
    # di = triton.runtime.driver.active.get_device_interface()

    @staticmethod
    def func_perf_test(func, args, warmup=1000, iterations=10000, quantiles=[0.2, 0.5, 0.8], **kwargs):
        assert kwargs == {} or kwargs is None
        return KernelPerfBench._method4(func, args, warmup, iterations, quantiles, **kwargs)
    
    @staticmethod
    def func_perf_test_mean(func, args, warmup=1000, iterations=10000, quantiles=[0.2, 0.5, 0.8], **kwargs):
        assert kwargs == {} or kwargs is None
        result = KernelPerfBench._method4(func, args, warmup, iterations, quantiles, **kwargs)
        return result['mean']
        
    @staticmethod
    def func_perf_test_median(func, args, warmup=1000, iterations=10000, quantiles=[0.2, 0.5, 0.8], **kwargs):
        assert kwargs == {} or kwargs is None
        result = KernelPerfBench._method4(func, args, warmup, iterations, quantiles, **kwargs)
        return result['median']
        
    @staticmethod
    def _method1(func, args, warmup=1000, iterations=10000, quantiles=[0.2, 0.5, 0.8], **kwargs):
        """方法1: 批量执行 + 单次事件测量"""
        # 预热
        for _ in range(warmup):
            func(*args, **kwargs)
        
        KernelPerfBench.di.synchronize()
        
        # 创建events
        start_event = KernelPerfBench.di.Event(enable_timing=True)
        end_event = KernelPerfBench.di.Event(enable_timing=True)
        
        # 测量时间
        KernelPerfBench.di.synchronize()
        start_event.record()
        for _ in range(iterations):
            func(*args, **kwargs)
        end_event.record()
        KernelPerfBench.di.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event)
        avg_time = elapsed_time / iterations
        
        return {
            'mean': avg_time,
            'total': elapsed_time,
            'var': 0.0,  # 批量测量无法计算方差
            'median': avg_time,
            **{f'q{int(q*100)}': avg_time for q in quantiles}
        }

    @staticmethod
    def _method2(func, args, warmup=1000, iterations=1000, quantiles=[0.2, 0.5, 0.8], **kwargs):
        """方法2: 逐次测量 + 最后同步"""
        # 预热
        for _ in range(warmup):
            func(*args, **kwargs)
        
        KernelPerfBench.di.synchronize()
        
        # 创建events
        start_events = [KernelPerfBench.di.Event(enable_timing=True) for _ in range(iterations)]
        end_events = [KernelPerfBench.di.Event(enable_timing=True) for _ in range(iterations)]
        
        # 测量时间
        KernelPerfBench.di.synchronize()
        for i in range(iterations):
            start_events[i].record()
            func(*args, **kwargs)
            end_events[i].record()
        KernelPerfBench.di.synchronize()
        
        times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_events, end_events)], dtype=torch.float)
        ret_quantiles = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        
        return {
            'mean': torch.mean(times).item(),
            'total': torch.sum(times).item(),
            'var': torch.var(times).item(),
            'median': torch.median(times).item(),
            **{f'q{int(q*100)}': ret_quantiles[i] for i, q in enumerate(quantiles)}
        }

    @staticmethod
    def _method3(func, args, warmup=1000, iterations=1000, quantiles=[0.2, 0.5, 0.8], **kwargs):
        """方法3: 逐次测量 + 逐次同步"""
        # 预热
        for _ in range(warmup):
            func(*args, **kwargs)
        
        KernelPerfBench.di.synchronize()
        
        # 创建events
        start_events = [KernelPerfBench.di.Event(enable_timing=True) for _ in range(iterations)]
        end_events = [KernelPerfBench.di.Event(enable_timing=True) for _ in range(iterations)]
        
        # 测量时间
        KernelPerfBench.di.synchronize()
        for i in range(iterations):
            start_events[i].record()
            func(*args, **kwargs)
            end_events[i].record()
            KernelPerfBench.di.synchronize()
        
        times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_events, end_events)], dtype=torch.float)
        ret_quantiles = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        
        return {
            'mean': torch.mean(times).item(),
            'total': torch.sum(times).item(),
            'var': torch.var(times).item(),
            'median': torch.median(times).item(),
            **{f'q{int(q*100)}': ret_quantiles[i] for i, q in enumerate(quantiles)}
        }

    @staticmethod
    def _method4(func, args, warmup=1000, iterations=1000, quantiles=[0.2, 0.5, 0.8], **kwargs):
        """方法4: 逐次测量 + 最后同步 + 缓存清理"""
        # 预热
        for _ in range(warmup):
            func(*args, **kwargs)
        
        KernelPerfBench.di.synchronize()
        
        # 创建events
        start_events = [KernelPerfBench.di.Event(enable_timing=True) for _ in range(iterations)]
        end_events = [KernelPerfBench.di.Event(enable_timing=True) for _ in range(iterations)]
        
        cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
        # 测量时间
        KernelPerfBench.di.synchronize()
        for i in range(iterations):
            cache.zero_()
            start_events[i].record()
            func(*args, **kwargs)
            end_events[i].record()
        KernelPerfBench.di.synchronize()
        
        times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_events, end_events)], dtype=torch.float)
        ret_quantiles = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        
        return {
            'mean': torch.mean(times).item(),
            'total': torch.sum(times).item(),
            'var': torch.var(times).item(),
            'median': torch.median(times).item(),
            **{f'q{int(q*100)}': ret_quantiles[i] for i, q in enumerate(quantiles)}
        }

    @staticmethod
    def _method5(func, args, warmup=1000, iterations=1000, quantiles=[0.2, 0.5, 0.8], **kwargs):
        """方法5: 逐次测量 + 逐次同步 + 缓存清理"""
        # 预热
        for _ in range(warmup):
            func(*args, **kwargs)
        
        KernelPerfBench.di.synchronize()
        
        # 创建events
        start_events = [KernelPerfBench.di.Event(enable_timing=True) for _ in range(iterations)]
        end_events = [KernelPerfBench.di.Event(enable_timing=True) for _ in range(iterations)]
        
        cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
        # 测量时间
        KernelPerfBench.di.synchronize()
        for i in range(iterations):
            cache.zero_()
            start_events[i].record()
            func(*args, **kwargs)
            end_events[i].record()
            KernelPerfBench.di.synchronize()
        
        times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_events, end_events)], dtype=torch.float)
        ret_quantiles = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        
        return {
            'mean': torch.mean(times).item(),
            'total': torch.sum(times).item(),
            'var': torch.var(times).item(),
            'median': torch.median(times).item(),
            **{f'q{int(q*100)}': ret_quantiles[i] for i, q in enumerate(quantiles)}
        }

def perf_test(func, args, warmup=1000, iterations=2000, quantiles=[0.2, 0.5, 0.8], **kwargs):
    return KernelPerfBench._method5(func, args, warmup, iterations, quantiles, **kwargs)