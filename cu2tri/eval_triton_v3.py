# -*- coding: utf-8 -*-
"""Triton内核评估模块 - 精简版"""

import os
import sys
import asyncio
import logging
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import gc

# 设置环境变量
os.environ["TORCH_USE_CUDA_DSA"] = "1" 
os.environ['TORCH_CUDA_ARCH_LIST'] = "Ada"

# 导入项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import torch
    from eval_.kernelbench_c.custom_mprunner import (
        triton_compare_torch_worker, triton_compare_cuda_worker, cuda_compare_torch_worker,
        triton_perf_worker, cuda_perf_worker, torch_perf_worker
    )
    from eval_.common.config import EvalConfig
    from eval_.common.verifier import CompareResult, PerformanceResult
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


class TritonKernelEvaluator:
    """Triton内核评估器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None, config: EvalConfig = EvalConfig()):
        self.logger = logger or self._create_logger()
        self.config = config
        
    def _create_logger(self) -> logging.Logger:
        """创建日志记录器"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        return logger
    
    def _setup_file_logging(self, base_dir: str, model_name: str, time_str: str, 
                           logfile_prefix: str = "", timestamp_log_dir: Optional[str] = None) -> str:
        """设置文件日志"""
        model_name_clean = model_name.replace('.', '_').replace('/', '_').replace('-', '_')
        
        if timestamp_log_dir:
            log_file = f"{timestamp_log_dir}/{logfile_prefix}eval.log"
        else:
            log_dir = f"{base_dir}/logs/{model_name_clean}"
            os.makedirs(log_dir, exist_ok=True)
            log_file = f"{log_dir}/{logfile_prefix}eval_{time_str}.log"
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 添加文件处理器
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        
        return log_file
    
    def _run_evaluation_core(self, triton_file: str, cuda_file: str, torch_ref_file: str,
                           config: EvalConfig, output_capture_file: Optional[str], timeout: int) -> Dict[str, Any]:
        """核心评估逻辑"""
        try:
            # 正确性测试
            self.logger.info("🧪 Running correctness tests...")
            
            # Triton vs PyTorch
            self.logger.info("🔍 Testing Triton vs PyTorch...")
            triton_torch_result = triton_compare_torch_worker(
                triton_file=triton_file,
                torch_ref_file=torch_ref_file,
                config=config,
                log_file_path=output_capture_file,
                timeout=timeout
            )
            if triton_torch_result.subproc_success:
                self.logger.info(f"Triton vs PyTorch diff: max_rel_err: {triton_torch_result.result.max_relative_error}, max_abs_err: {triton_torch_result.result.max_absolute_error}")
            else:
                self.logger.info(f"Triton vs PyTorch failed - error: {triton_torch_result.error}, traceback: {triton_torch_result.traceback}, output_capture: {triton_torch_result.output_capture}")
            
            if not triton_torch_result.subproc_success:
                return {
                    "success": False,
                    "error": f"Triton vs PyTorch test failed: {triton_torch_result.error}",
                    "subprocess_failed": True
                }
            
            triton_torch_compare: CompareResult = triton_torch_result.result
            
            # Triton vs CUDA
            self.logger.info("🔍 Testing Triton vs CUDA...")
            triton_cuda_result = triton_compare_cuda_worker(
                triton_file=triton_file,
                cuda_ref_file=cuda_file,
                torch_ref_file=torch_ref_file,
                config=config,
                log_file_path=output_capture_file,
                timeout=timeout
            )
            if triton_cuda_result.subproc_success:
                self.logger.info(f"Triton vs CUDA diff: max_rel_err: {triton_cuda_result.result.max_relative_error}, max_abs_err: {triton_cuda_result.result.max_absolute_error}")
            else:
                self.logger.info(f"Triton vs CUDA failed - error: {triton_cuda_result.error}, traceback: {triton_cuda_result.traceback}, output_capture: {triton_cuda_result.output_capture}")
            
            if not triton_cuda_result.subproc_success:
                return {
                    "success": False,
                    "error": f"Triton vs CUDA test failed: {triton_cuda_result.error}",
                    "subprocess_failed": True
                }
            
            triton_cuda_compare: CompareResult = triton_cuda_result.result
            
            # CUDA vs PyTorch
            self.logger.info("🔍 Testing CUDA vs PyTorch...")
            cuda_torch_result = cuda_compare_torch_worker(
                cuda_file=cuda_file,
                torch_ref_file=torch_ref_file,
                config=config,
                log_file_path=output_capture_file,
                timeout=timeout
            )
            if cuda_torch_result.subproc_success:
                self.logger.info(f"CUDA vs PyTorch diff: max_rel_err: {cuda_torch_result.result.max_relative_error}, max_abs_err: {cuda_torch_result.result.max_absolute_error}")
            else:
                self.logger.info(f"CUDA vs PyTorch failed - error: {cuda_torch_result.error}, traceback: {cuda_torch_result.traceback}, output_capture: {cuda_torch_result.output_capture}")
            
            if not cuda_torch_result.subproc_success:
                return {
                    "success": False,
                    "error": f"CUDA vs PyTorch test failed: {cuda_torch_result.error}",
                    "subprocess_failed": True
                }
            
            cuda_torch_compare: CompareResult = cuda_torch_result.result
            
            self.logger.info(f"   Triton vs PyTorch: {'✅' if triton_torch_compare.overall_match else '❌'}")
            self.logger.info(f"   Triton vs CUDA: {'✅' if triton_cuda_compare.overall_match else '❌'}")
            self.logger.info(f"   CUDA vs PyTorch: {'✅' if cuda_torch_compare.overall_match else '❌'}")
            
            correctness = {
                "triton_torch_match": triton_torch_compare.overall_match,
                "triton_cuda_match": triton_cuda_compare.overall_match,
                "cuda_torch_match": cuda_torch_compare.overall_match
            }
            
            if not triton_cuda_compare.overall_match:
                return {
                    "success": False,
                    "error": "Function test failed: result mismatch",
                    "correctness": correctness
                }
            
            # 性能测试
            self.logger.info("⏱️  Running performance tests...")
            
            # Triton性能测试
            self.logger.info("🔥 Testing Triton performance...")
            triton_perf_result = triton_perf_worker(
                triton_file=triton_file,
                torch_ref_file=torch_ref_file,
                config=config,
                log_file_path=output_capture_file
            )
            
            if hasattr(triton_perf_result, "result") and hasattr(triton_perf_result.result, "perf_time_ms"):
                self.logger.info(f"Triton perf: {triton_perf_result.result.perf_time_ms}")
            else:
                self.logger.info(f"Triton perf: {triton_perf_result}")
            
            if not triton_perf_result.subproc_success:
                return {
                    "success": False,
                    "error": f"Triton performance test failed: {triton_perf_result.error}",
                    "correctness": correctness
                }
            
            triton_perf: PerformanceResult = triton_perf_result.result
            triton_time = triton_perf.perf_time_ms
            
            # CUDA性能测试
            self.logger.info("🔥 Testing CUDA performance...")
            cuda_perf_result = cuda_perf_worker(
                cuda_file=cuda_file,
                torch_ref_file=torch_ref_file,
                config=config,
                log_file_path=output_capture_file
            )
            
            if hasattr(cuda_perf_result, "result") and hasattr(cuda_perf_result.result, "perf_time_ms"):
                self.logger.info(f"CUDA perf: {cuda_perf_result.result.perf_time_ms}")
            else:
                self.logger.info(f"CUDA perf: {cuda_perf_result}")
            
            if not cuda_perf_result.subproc_success:
                return {
                    "success": False,
                    "error": f"CUDA performance test failed: {cuda_perf_result}",
                    "correctness": correctness
                }
            
            cuda_perf: PerformanceResult = cuda_perf_result.result
            cuda_time = cuda_perf.perf_time_ms
            
            # PyTorch性能测试
            self.logger.info("🔥 Testing PyTorch performance...")
            torch_perf_result = torch_perf_worker(
                torch_file=torch_ref_file,
                config=config,
                log_file_path=output_capture_file
            )
            
            if hasattr(torch_perf_result, "result") and hasattr(torch_perf_result.result, "perf_time_ms"):
                self.logger.info(f"PyTorch perf: {torch_perf_result.result.perf_time_ms}")
            else:
                self.logger.info(f"PyTorch perf: {torch_perf_result}")
            
            if not torch_perf_result.subproc_success:
                return {
                    "success": False,
                    "error": f"PyTorch performance test failed: {torch_perf_result}",
                    "correctness": correctness
                }
            
            torch_perf: PerformanceResult = torch_perf_result.result
            torch_time = torch_perf.perf_time_ms
            
            # 获取设备信息
            device_name = "Unknown"
            device_id = 0
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(device_id)
            
            return {
                "success": True,
                "performance": {
                    "triton_time": triton_time,
                    "cuda_time": cuda_time,
                    "torch_time": torch_time,
                    "triton_cuda_speedup": cuda_time / triton_time if triton_time > 0 else 0.0,
                    "triton_torch_speedup": torch_time / triton_time if triton_time > 0 else 0.0
                },
                "correctness": correctness,
                "device_info": {"device_name": device_name, "device_id": device_id},
                "output_capture": (triton_torch_compare.output_capture + 
                                 triton_cuda_compare.output_capture + 
                                 cuda_torch_compare.output_capture +
                                 triton_perf.output_capture +
                                 cuda_perf.output_capture +
                                 torch_perf.output_capture)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation error: {e}")
            return {"success": False, "error": str(e)}

    async def evaluate_triton_kernel(self, base_dir: str, kernel_name: str = None,
                                   model_name: str = "triton_test", time_str: Optional[str] = None,
                                   logfile_prefix: str = "", timestamp_log_dir: Optional[str] = None,
                                   return_result: bool = False, timeout: int = 300,
                                   capture_output: bool = True) -> Optional[Dict[str, Any]]:
        """评估Triton内核"""
        
        if time_str is None:
            time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 确定文件路径
        if timestamp_log_dir and logfile_prefix.startswith("round"):
            round_num = logfile_prefix.replace("round", "").replace("_", "")
            triton_file = f"{timestamp_log_dir}/triton_round{round_num}.py"
        elif timestamp_log_dir:
            triton_file = f"{timestamp_log_dir}/triton_ref.py"
        else:
            triton_file = f"{base_dir}/triton_ref.py"
        
        cuda_file = f"{base_dir}/cuda_ref.cu"
        torch_ref_file = f"{base_dir}/torch_ref.py"
        
        # 设置日志
        log_file = self._setup_file_logging(base_dir, model_name, time_str, logfile_prefix, timestamp_log_dir)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Test Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'='*60}")
        
        try:
            # 检查文件存在性
            for file_path, desc in [(triton_file, "Triton"), (cuda_file, "CUDA"), (torch_ref_file, "PyTorch")]:
                if not os.path.exists(file_path):
                    error_msg = f"{desc} file not found: {file_path}"
                    self.logger.error(f"❌ {error_msg}")
                    return {"success": False, "error": error_msg} if return_result else None
            kernel_name = Path(base_dir).name
            # 准备配置
            config = EvalConfig(
                cuda_kernel_name=kernel_name,
                build_dir=f"{base_dir}/build",
                atol=self.config.atol,
                rtol=self.config.rtol,
                subproc_timeout=timeout
            )
            
            output_capture_file = f"{os.path.dirname(log_file)}/subprocess_output_{time_str}.log" if capture_output else None
            
            self.logger.info(f"🚀 Evaluating files:")
            self.logger.info(f"   Triton: {triton_file}")
            self.logger.info(f"   CUDA: {cuda_file}")
            self.logger.info(f"   PyTorch: {torch_ref_file}")
            
            # 执行评估
            result = await asyncio.to_thread(
                self._run_evaluation_core, triton_file, cuda_file, torch_ref_file, config, output_capture_file, timeout
            )
            
            # 记录结果
            if result.get("success"):
                self.logger.info("✅ Evaluation completed successfully")
                if "performance" in result:
                    perf = result["performance"]
                    self.logger.info(f"\n📈 Performance Results:")
                    self.logger.info(f"   Triton: {perf['triton_time']:.3f} ms")
                    self.logger.info(f"   CUDA: {perf['cuda_time']:.3f} ms")
                    self.logger.info(f"   PyTorch: {perf['torch_time']:.3f} ms")
                    self.logger.info(f"   Speedup: {perf['triton_cuda_speedup']:.2f}x vs CUDA")
            else:
                self.logger.error(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}")
            
            self.logger.info(f"\n✅ Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"{'='*60}\n")
            
            return result if return_result else None
            
        except Exception as e:
            self.logger.error(f"❌ Main process error: {e}")
            return {"success": False, "error": str(e)} if return_result else None
        
        finally:
            # 清理GPU内存
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass


# 测试函数
async def test_square_matrix_multiplication():
    """测试方阵乘法"""
    config = EvalConfig(atol=0.1, rtol=0.1)
    evaluator = TritonKernelEvaluator(config=config)
    
    test_dir = "outputs/tests/1_Square_matrix_multiplication_"
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    print(f"🚀 Testing: {test_dir}")
    result = await evaluator.evaluate_triton_kernel(
        base_dir=test_dir,
        model_name="square_matrix_multiplication",
        return_result=True,
        timeout=180
    )
    
    if result:
        print(f"\n📊 Result: {result['success']}")
        if result.get('performance'):
            perf = result['performance']
            print(f"Performance: Triton {perf['triton_time']:.3f}ms, CUDA {perf['cuda_time']:.3f}ms, Speedup {perf['triton_cuda_speedup']:.2f}x")
        if not result['success']:
            print(f"❌ Error: {result.get('error')}")


async def main():
    """主函数"""
    print("🎯 Triton Kernel Evaluator V3 - Simplified")
    print("="*50)
    await test_square_matrix_multiplication()


if __name__ == "__main__":
    asyncio.run(main())
