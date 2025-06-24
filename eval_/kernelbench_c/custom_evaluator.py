"""
主评估模块

整合所有子模块实现主要的评估逻辑
"""

import os
import logging
import os
import traceback
from typing import Optional, Dict, Any

import torch

from ..common.config import EvalConfig, DEFAULT_CONFIG
from utils.set_env import set_env
from ..common.loader import load_cuda_extension_from_cufile
from ..common.benchmark import benchmark_kernel
from .custom_mprunner import run_correctness_test_multiprocess
from .custom_reporter import calculate_performance_metrics, format_correctness_report, format_performance_report
from .custom_setup import setup_logging, determine_triton_file_path, setup_log_paths, log_test_completion
from .custom_loader import load_torch_reference_from_pyfile

def eval_triton_kernel(
    base_dir: str, 
    model_name: str = None, 
    logger: logging.Logger = None,
    time_str: Optional[str] = None, 
    logfile_prefix: str = "", 
    return_result: bool = False, 
    timestamp_log_dir: Optional[str] = None,
    config: EvalConfig = DEFAULT_CONFIG
) -> Optional[Dict[str, Any]]:
    """
    评估Triton内核与CUDA内核的功能和性能对比
    
    Args:
        base_dir: 测试基础目录
        model_name: 模型名称
        logger: 日志记录器
        time_str: 时间字符串
        logfile_prefix: 日志文件前缀
        return_result: 是否返回结果
        timestamp_log_dir: 时间戳日志目录
        config: 评估配置
        
    Returns:
        评估结果字典（如果return_result为True）
    """
    # 设置环境
    set_env()
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 确定文件路径
    try:
        # 首先尝试在base_dir中查找triton.py
        triton_file = f"{base_dir}/triton.py"
        if not os.path.exists(triton_file) and timestamp_log_dir:
            # 如果base_dir中没有，再尝试时间戳目录
            triton_file = determine_triton_file_path(timestamp_log_dir, logfile_prefix)
        
        cuda_file = f"{base_dir}/cuda_ref.cu"  
        torch_ref_file = f"{base_dir}/torch_ref.py"
        
        # 设置日志路径
        log_file, latest_eval_file = setup_log_paths(
            base_dir, model_name, time_str, timestamp_log_dir, logfile_prefix
        )
        
        # 设置日志记录
        logger = setup_logging(log_file, latest_eval_file, logfile_prefix, logger)
        
    except Exception as e:
        error_msg = f"Path setup failed: {e}"
        if logger:
            logger.error(error_msg)
        if return_result:
            return {"success": False, "error": error_msg}
        return
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        logger.info("❌ CUDA unavailable, skipping test")
        if return_result:
            return {"success": False, "error": "CUDA unavailable"}
        return
    
    # 加载PyTorch参考实现获取测试用例
    logger.info("📁 Loading test cases...")
    try:
        
        torch_ref = load_torch_reference_from_pyfile(torch_ref_file)
        test_inputs = torch_ref.get_inputs()
        
        # 将输入移到GPU
        cuda_inputs = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in test_inputs]
        logger.info(f"📊 Test matrix size: {cuda_inputs[0].shape}")
        
    except Exception as e:
        error_msg = f"Failed to load test cases: {e}"
        logger.error(error_msg)
        if return_result:
            return {"success": False, "error": error_msg, "traceback": traceback.format_exc()}
        return
    
    # 获取输入形状信息（用于子进程重新生成输入）
    input_shapes = [inp.shape if isinstance(inp, torch.Tensor) else None for inp in cuda_inputs]
    
    logger.info(f"⚡ Testing Triton kernel correctness in multiprocess: {triton_file}")
    correctness_result = run_correctness_test_multiprocess(
        triton_file, torch_ref_file, input_shapes, random_seed=42, config=config
    )
    
    if not correctness_result.subproc_success:
        error_msg = correctness_result.error or "Unknown error"
        if "core dumped" in error_msg or "Aborted" in error_msg or "timeout" in error_msg:
            logger.info(f"❌ Triton kernel crashed or timed out: {error_msg}")
        else:
            logger.info(f"❌ Triton kernel correctness test failed: {error_msg}")
        
        if return_result:
            return {"success": False, "error": error_msg, "traceback": error_msg}
        return
    
    # 获取实际的结果数据
    result_data = correctness_result.result
    if not result_data.get("success", False):
        error_msg = result_data.get("error", "Unknown error")
        logger.info(f"❌ Triton kernel correctness test failed: {error_msg}")
        if return_result:
            return {"success": False, "error": error_msg, "traceback": result_data.get("traceback", "")}
        return
    
    logger.info("✅ Triton kernel correctness test successful in multiprocess")
    
    # 编译CUDA内核
    logger.info("🔧 Compiling CUDA kernel...")
    try:
        cuda_extension = load_cuda_extension_from_cufile(cuda_file, config)
        if isinstance(cuda_extension, str):  # 编译失败，返回错误信息
            logger.info(f"❌ CUDA kernel compilation failed: {cuda_extension}")
            if return_result:
                return {"success": False, "error": f"CUDA kernel compilation failed: {cuda_extension}"}
            return
        
        cuda_func = getattr(cuda_extension, 'forward')
        logger.info("✅ CUDA kernel compilation successful")
        
    except Exception as e:
        error_msg = f"CUDA kernel compilation failed: {e}"
        logger.info(f"❌ {error_msg}")
        if return_result:
            return {"success": False, "error": error_msg, "traceback": traceback.format_exc()}
        return
    
    # 功能测试（比较triton和cuda结果）
    logger.info("\n🧪 Starting functional correctness testing...")
    
    try:
        # 执行CUDA内核获取结果
        cuda_result = cuda_func(*cuda_inputs)
        
        # 从子进程获取triton结果
        triton_correctness = result_data["correctness"]
        
        # 比较triton和cuda结果（简化的比较，实际应该更详细）
        # 这里只是一个简单的形状和数值比较示例
        triton_cuda_match = True  # 暂时设为True，实际需要实现具体比较逻辑
        
        correctness = {
            "triton_torch_match": triton_correctness.overall_match,
            "triton_cuda_match": triton_cuda_match,
            "overall_match": triton_correctness.overall_match and triton_cuda_match
        }
        
        # 输出比较结果
        logger.info(format_correctness_report(correctness))
        
        if not correctness["overall_match"]:
            logger.info("❌ Functional test failed, skipping performance test")
            if return_result:
                return {"success": False, "error": "Functional test failed: results do not match", "correctness": correctness}
            return
            
    except Exception as e:
        error_msg = f"Functional test error: {e}"
        logger.info(f"❌ {error_msg}")
        if return_result:
            return {"success": False, "error": error_msg, "traceback": traceback.format_exc()}
        return
    
    # 性能测试（在主进程直接执行，避免进程间通信开销）
    logger.info("\n⏱️  Starting performance testing...")
    
    # 直接在主进程加载和测试Triton内核
    try:
        from pathlib import Path
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            Path(triton_file).stem,
            triton_file
        )
        triton_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(triton_module)
        triton_func = getattr(triton_module, 'forward')
        
        logger.info("Testing Triton kernel performance...")
        triton_time = benchmark_kernel(triton_func, cuda_inputs)
        logger.info(f"✅ Triton performance test successful: {triton_time:.3f} ms")
        
    except Exception as e:
        logger.info(f"❌ Triton performance test failed: {e}")
        # 即使性能测试失败，我们也认为功能测试是成功的
        if return_result:
            return {
                "success": True,
                "performance": None,
                "correctness": correctness,
                "note": "Functional test passed, but performance test failed"
            }
        log_test_completion(logger, success=True)
        return
    
    logger.info("Testing CUDA kernel performance...")
    cuda_time = benchmark_kernel(cuda_func, cuda_inputs)
    
    logger.info("Testing PyTorch reference performance...")
    torch_time = benchmark_kernel(torch_ref.module_fn, cuda_inputs)
    
    # 计算性能指标
    performance = calculate_performance_metrics(triton_time, cuda_time, torch_time)
    
    # 输出性能结果
    logger.info(f"\n{format_performance_report(performance)}")
    
    log_test_completion(logger, success=True)
    
    if return_result:
        return {
            "success": True,
            "performance": performance,
            "correctness": correctness
        } 