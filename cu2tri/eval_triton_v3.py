# -*- coding: utf-8 -*-
"""Triton Kernel Evaluation Module - Optimized Version"""

import os
import sys
import asyncio
import logging
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Callable
import gc

# Set environment variables
os.environ["TORCH_USE_CUDA_DSA"] = "1" 
os.environ['TORCH_CUDA_ARCH_LIST'] = "Ada"

# Add project root to path
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
    """Triton Kernel Evaluator"""
    
    def __init__(self, logger: Optional[logging.Logger] = None, config: EvalConfig = EvalConfig()):
        self.logger = logger or self._create_logger()
        self.config = config
        
    def _create_logger(self) -> logging.Logger:
        """Create logger"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-5s | %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _setup_file_logging(self, base_dir: Path, model_name: str, time_str: str, 
                           logfile_prefix: str = "", timestamp_log_dir: Optional[Path] = None) -> str:
        """Setup file logging"""
        model_name_clean = model_name.replace('.', '_').replace('/', '_').replace('-', '_')
        
        if timestamp_log_dir:
            log_file = timestamp_log_dir / f"{logfile_prefix}eval.log"
        else:
            log_dir = base_dir / "logs" / model_name_clean
            os.makedirs(log_dir, exist_ok=True)
            log_file = log_dir / f"{logfile_prefix}eval_{time_str}.log"
        
        os.makedirs(log_file.parent, exist_ok=True)
        
        # Add file handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-5s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        
        return log_file
    
    def _run_compare_test(self, worker_func: Callable, test_name: str, 
                         config: EvalConfig, output_capture_file: Optional[str], 
                         timeout: int, **kwargs) -> Tuple[bool, CompareResult, str]:
        """Run comparison test with generic function"""
        self.logger.info(f"🔍 {test_name}...")
        
        result = worker_func(
            config=config,
            log_file_path=output_capture_file,
            timeout=timeout,
            **kwargs
        )
        
        if result.subproc_success:
            compare_result: CompareResult = result.result
            self.logger.info(
                f"    ✓ {test_name}: "
                f"max_rel_err={compare_result.max_relative_error:.3f}, "
                f"max_abs_err={compare_result.max_absolute_error:.6f}"
            )
            return True, compare_result, ""
        else:
            error_msg = f"{test_name} failed - {result.error}"
            self.logger.error(f"    ❌ {error_msg}")
            if result.traceback:
                self.logger.debug(f"    Traceback: {result.traceback}")
            return False, None, error_msg
    
    def _run_perf_test(self, worker_func: Callable, test_name: str,
                      config: EvalConfig, output_capture_file: Optional[str],
                      **kwargs) -> Tuple[bool, PerformanceResult, str]:
        """Run performance test with generic function"""
        self.logger.info(f"🚀 {test_name}...")
        
        result = worker_func(
            config=config,
            log_file_path=output_capture_file,
            **kwargs
        )
        
        if result.subproc_success:
            perf_result: PerformanceResult = result.result
            self.logger.info(f"    ✓ {test_name}: {perf_result.perf_time_ms:.3f} ms")
            return True, perf_result, ""
        else:
            error_msg = f"{test_name} failed - {result.error}"
            self.logger.error(f"    ❌ {error_msg}")
            if hasattr(result, 'traceback') and result.traceback:
                self.logger.debug(f"    Traceback: {result.traceback}")
            return False, None, error_msg
    
    def _run_evaluation_core(self, triton_file: str, cuda_file: str, torch_ref_file: str,
                           config: EvalConfig, output_capture_file: Optional[str], timeout: int) -> Dict[str, Any]:
        """Core evaluation logic"""
        try:
            # ═══════════════ Correctness Testing ═══════════════
            self.logger.info("🧪 Correctness Testing")
            self.logger.info("─" * 50)
            
            # Triton vs PyTorch
            success, triton_torch_compare, error = self._run_compare_test(
                triton_compare_torch_worker, "Triton vs PyTorch",
                config, output_capture_file, timeout,
                triton_file=triton_file, torch_ref_file=torch_ref_file
            )
            if not success:
                return {"success": False, "error": error, "subprocess_failed": True}
            
            # Triton vs CUDA
            success, triton_cuda_compare, error = self._run_compare_test(
                triton_compare_cuda_worker, "Triton vs CUDA",
                config, output_capture_file, timeout,
                triton_file=triton_file, cuda_ref_file=cuda_file,
                torch_ref_file=torch_ref_file
            )
            if not success:
                return {"success": False, "error": error, "subprocess_failed": True}
            
            # CUDA vs PyTorch
            success, cuda_torch_compare, error = self._run_compare_test(
                cuda_compare_torch_worker, "CUDA vs PyTorch",
                config, output_capture_file, timeout,
                cuda_file=cuda_file, torch_ref_file=torch_ref_file
            )
            if not success:
                return {"success": False, "error": error, "subprocess_failed": True}
            
            # Output correctness summary
            self.logger.info("─" * 50)
            self.logger.info("📋 Correctness Summary:")
            self.logger.info(f"    Triton vs PyTorch: {'✅ PASS' if triton_torch_compare.overall_match else '❌ FAIL'}")
            self.logger.info(f"    Triton vs CUDA:    {'✅ PASS' if triton_cuda_compare.overall_match else '❌ FAIL'}")
            self.logger.info(f"    CUDA vs PyTorch:   {'✅ PASS' if cuda_torch_compare.overall_match else '❌ FAIL'}")
            
            correctness = {
                "triton_torch_match": triton_torch_compare.overall_match,
                "triton_cuda_match": triton_cuda_compare.overall_match,
                "cuda_torch_match": cuda_torch_compare.overall_match
            }
            
            if not triton_cuda_compare.overall_match:
                return {
                    "success": False,
                    "error": "Correctness test failed: result mismatch",
                    "correctness": correctness
                }
            
            # ═══════════════ Performance Testing ═══════════════
            self.logger.info("")
            self.logger.info("⚡ Performance Testing")
            self.logger.info("─" * 50)
            
            # Triton performance test
            success, triton_perf, error = self._run_perf_test(
                triton_perf_worker, "Triton Performance",
                config, output_capture_file,
                triton_file=triton_file, torch_ref_file=torch_ref_file
            )
            if not success:
                return {"success": False, "error": error, "correctness": correctness}
            
            # CUDA performance test
            success, cuda_perf, error = self._run_perf_test(
                cuda_perf_worker, "CUDA Performance",
                config, output_capture_file,
                cuda_file=cuda_file, torch_ref_file=torch_ref_file
            )
            if not success:
                return {"success": False, "error": error, "correctness": correctness}
            
            # PyTorch performance test
            success, torch_perf, error = self._run_perf_test(
                torch_perf_worker, "PyTorch Performance",
                config, output_capture_file,
                torch_file=torch_ref_file
            )
            if not success:
                return {"success": False, "error": error, "correctness": correctness}
            
            # Performance summary
            triton_time = triton_perf.perf_time_ms
            cuda_time = cuda_perf.perf_time_ms
            torch_time = torch_perf.perf_time_ms
            
            triton_cuda_speedup = cuda_time / triton_time if triton_time > 0 else 0.0
            triton_torch_speedup = torch_time / triton_time if triton_time > 0 else 0.0
            
            self.logger.info("─" * 50)
            self.logger.info("📊 Performance Summary:")
            self.logger.info(f"    Triton:   {triton_time:8.3f} ms")
            self.logger.info(f"    CUDA:     {cuda_time:8.3f} ms")
            self.logger.info(f"    PyTorch:  {torch_time:8.3f} ms")
            self.logger.info("─" * 30)
            self.logger.info(f"    Speedup (vs CUDA):    {triton_cuda_speedup:6.2f}x")
            self.logger.info(f"    Speedup (vs PyTorch): {triton_torch_speedup:6.2f}x")
            
            # Get device info
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
                    "triton_cuda_speedup": triton_cuda_speedup,
                    "triton_torch_speedup": triton_torch_speedup
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

    async def evaluate_triton_kernel(self, base_dir: Path, kernel_name: str = None,
                                   model_name: str = "triton_test", time_str: Optional[str] = None,
                                   logfile_prefix: str = "", timestamp_log_dir: Optional[Path] = None,
                                   return_result: bool = False, timeout: int = 300,
                                   capture_output: bool = True) -> Optional[Dict[str, Any]]:
        """Evaluate Triton kernel"""
        
        if time_str is None:
            time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Determine file paths
        if timestamp_log_dir and logfile_prefix.startswith("round"):
            round_num = logfile_prefix.replace("round", "").replace("_", "")
            triton_file = timestamp_log_dir / f"triton_round{round_num}.py"
        elif timestamp_log_dir:
            triton_file = timestamp_log_dir / "triton_ref.py"
        else:
            triton_file = base_dir / "triton_ref.py"
        
        cuda_file = base_dir / "cuda_ref.cu"
        torch_ref_file = base_dir / "torch_ref.py"
        
        # Setup logging
        log_file = self._setup_file_logging(base_dir, model_name, time_str, logfile_prefix, timestamp_log_dir)
        
        self.logger.info("")
        self.logger.info("╔" + "═" * 58 + "╗")
        self.logger.info("║" + f"{'Triton Kernel Evaluation Started':^58}" + "║")
        self.logger.info("║" + f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^58}" + "║")
        self.logger.info("╚" + "═" * 58 + "╝")
        
        try:
            # Check file existence
            for file_path, desc in [(triton_file, "Triton"), (cuda_file, "CUDA"), (torch_ref_file, "PyTorch")]:
                if not os.path.exists(file_path):
                    error_msg = f"{desc} file not found: {file_path}"
                    self.logger.error(f"❌ {error_msg}")
                    return {"success": False, "error": error_msg} if return_result else None
            
            kernel_name = Path(base_dir).name
            
            # Prepare configuration
            config = EvalConfig(
                cuda_kernel_name=kernel_name,
                build_dir=f"{base_dir}/build",
                atol=self.config.atol,
                rtol=self.config.rtol,
                subproc_timeout=timeout
            )
            
            output_capture_file = log_file.parent / f"subprocess_output_{time_str}.log" if capture_output else None
            
            self.logger.info("📁 File Paths:")
            self.logger.info(f"    Triton:   {triton_file}")
            self.logger.info(f"    CUDA:     {cuda_file}")
            self.logger.info(f"    PyTorch:  {torch_ref_file}")
            self.logger.info("")
            
            # Execute evaluation
            result = await asyncio.to_thread(
                self._run_evaluation_core, str(triton_file), str(cuda_file), str(torch_ref_file), config, output_capture_file, timeout
            )
            
            # Log results
            self.logger.info("")
            if result.get("success"):
                self.logger.info("╔" + "═" * 58 + "╗")
                self.logger.info("║" + f"{'🎉 Evaluation Completed Successfully!':^56}" + "║")
                self.logger.info("╚" + "═" * 58 + "╝")
            else:
                self.logger.error("╔" + "═" * 58 + "╗")
                self.logger.error("║" + f"{'❌ Evaluation Failed':^58}" + "║")
                self.logger.error("║" + f"{result.get('error', 'Unknown error')[:54]:^58}" + "║")
                self.logger.error("╚" + "═" * 58 + "╝")
            
            self.logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("")
            
            return result if return_result else None
            
        except Exception as e:
            self.logger.error(f"❌ Main process error: {e}")
            return {"success": False, "error": str(e)} if return_result else None
        
        finally:
            # Clean up GPU memory
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass


# Test functions
async def test_square_matrix_multiplication():
    """Test square matrix multiplication"""
    config = EvalConfig(atol=0.1, rtol=0.1)
    evaluator = TritonKernelEvaluator(config=config)
    from utils.set_env import PROJECT_ROOT
    test_dir = PROJECT_ROOT / "cu2tri" / "outputs" / "tests" / "1_Square_matrix_multiplication_"
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    print(f"🚀 Testing directory: {test_dir}")
    result = await evaluator.evaluate_triton_kernel(
        base_dir=test_dir,
        model_name="square_matrix_multiplication",
        return_result=True,
        timeout=180
    )
    
    if result:
        print(f"\n📊 Final result: {'Success' if result['success'] else 'Failed'}")
        if result.get('performance'):
            perf = result['performance']
            print(f"Performance: Triton {perf['triton_time']:.3f}ms, CUDA {perf['cuda_time']:.3f}ms, Speedup {perf['triton_cuda_speedup']:.2f}x")
        if not result['success']:
            print(f"❌ Error: {result.get('error')}")


async def main():
    """Main function"""
    print("🎯 Triton Kernel Evaluator V3 - Optimized")
    print("=" * 50)
    await test_square_matrix_multiplication()


if __name__ == "__main__":
    asyncio.run(main())
