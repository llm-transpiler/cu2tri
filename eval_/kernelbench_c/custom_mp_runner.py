"""
基于multiprocessing的子进程运行器模块

使用multiprocessing在子进程中运行Triton测试，防止core dump影响主进程
相比subprocess方式，避免了临时文件生成和字符串脚本的开销
"""

import os
import signal
import multiprocessing
from typing import Dict, Any, List
from pathlib import Path
import importlib.util
import traceback
from pydantic import BaseModel
import torch

# 设置多进程启动方法为 spawn 以支持 CUDA
multiprocessing.set_start_method('spawn', force=True)

from ..common.config import EvalConfig, DEFAULT_CONFIG
from ..common.reporter import CorrectnessResult, _compare_tensor_results, PerformanceResult
from ..common.loader import _load_pyfile_module
from ..common.loader import _load_pyfile_module_attr
from ..common.mprunner import mp_run


def _correctness_worker(triton_file, torch_ref_file, input_shapes, random_seed, config, result_queue):
    """Worker function for correctness testing"""
    try:
        # 设置随机种子确保可重现性
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        
        # 加载torch参考实现
        torch_ref = _load_torch_reference(torch_ref_file)
        
        # 生成输入数据
        test_inputs = torch_ref.get_inputs()
        cuda_inputs = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in test_inputs]
        
        # 执行torch参考
        torch_result = torch_ref.module_fn(*cuda_inputs)
        
        # 加载并执行Triton内核
        triton_func = _load_triton_kernel(triton_file)
        triton_result = triton_func(*cuda_inputs)
        
        # 计算匹配度
        match_result = _compare_tensor_results(triton_result, torch_result, config)
        
        result_queue.put({
            "success": True,
            "correctness": match_result,
            "triton_shape": list(triton_result.shape),
            "triton_dtype": str(triton_result.dtype),
            "torch_shape": list(torch_result.shape),
            "torch_dtype": str(torch_result.dtype)
        })
        
    except Exception as e:
        result_queue.put({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def run_correctness_test_multiprocess(
    triton_file: str,
    torch_ref_file: str,
    input_shapes: List[tuple],
    random_seed: int = 42,
    config: EvalConfig = DEFAULT_CONFIG
) -> CorrectnessResult:
    """
    在子进程中运行正确性测试
    
    Args:
        triton_file: Triton内核文件路径
        torch_ref_file: PyTorch参考实现文件路径
        input_shapes: 输入张量的形状列表
        random_seed: 随机种子
        config: 测试配置
        
    Returns:
        测试结果字典
    """
    return mp_run(_correctness_worker, (triton_file, torch_ref_file, input_shapes, random_seed, config), timeout=config.subproc_timeout)

def _performance_worker(triton_file, torch_ref_file, input_shapes, random_seed, config, result_queue):
    """Worker function for performance testing"""
    try:
        # 设置随机种子确保可重现性
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        
        # 加载torch参考实现
        torch_ref = _load_torch_reference(torch_ref_file)
        
        # 生成输入数据
        test_inputs = torch_ref.get_inputs()
        cuda_inputs = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in test_inputs]
        
        # 加载Triton内核
        triton_func = _load_triton_kernel(triton_file)
        from ..common.benchmark import _simple_perf
        # 性能测试
        triton_time = _simple_perf(triton_func, cuda_inputs, warmup=config.warmup_runs, iterations=config.test_runs, get_mean=True)
        
        result_queue.put({
            "success": True,
            "perf_time_ms": triton_time
        })
        
    except Exception as e:
        result_queue.put({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def run_performance_test_multiprocess(
    triton_file: str,
    torch_ref_file: str,
    input_shapes: List[tuple],
    random_seed: int = 42,
    config: EvalConfig = DEFAULT_CONFIG
) -> PerformanceResult:
    return mp_run(_performance_worker, (triton_file, torch_ref_file, input_shapes, random_seed, config), timeout=config.subproc_timeout)

def _load_torch_reference(torch_ref_file: str):
    return _load_pyfile_module(torch_ref_file)


def _load_triton_kernel(triton_file: str):
    return _load_pyfile_module_attr(triton_file, "forward")
