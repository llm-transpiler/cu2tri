# -*- coding: utf-8 -*-
"""CUDA Kernel Evaluation Module - Simplified Version"""

import os
import sys
import asyncio
import logging
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Callable
import gc
import json

from utils.set_env import set_env
set_env()
# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
from utils.set_env import PROJECT_ROOT
from eval_.common.config import DEFAULT_CONFIG
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
    from eval_.kernelbench_c.custom_mprunner import (
        cuda_compare_torch_worker, cuda_perf_worker, torch_perf_worker
    )
    from eval_.common.config import EvalConfig, DEFAULT_RANDOM_SEED
    from eval_.common.verifier import CompareResult, PerformanceResult
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

import requests
import time

API_BASE_URL = "http://localhost:8081"

def submit_task(task_data: Dict[str, Any]) -> str:
    """Submit a task and return task ID"""
    response = requests.post(f"{API_BASE_URL}/tasks/submit", json=task_data)
    if response.status_code == 200:
        result = response.json()
        return result["task_id"]
    else:
        raise Exception(f"Failed to submit task: {response.text}")

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get task status"""
    response = requests.get(f"{API_BASE_URL}/tasks/{task_id}")
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get task status: {response.text}")

def wait_for_task_completion(task_id: str, timeout: int = DEFAULT_CONFIG().subproc_timeout) -> Dict[str, Any]:
    """Wait for task to complete"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = get_task_status(task_id)
        if status["status"] in ["completed", "failed", "cancelled"]:
            return status
        time.sleep(2)
    raise Exception(f"Task {task_id} did not complete within {timeout} seconds")

class CUDAKernelEvaluator:
    """CUDA Kernel Evaluator - Only CUDA vs PyTorch comparison"""
    
    def __init__(self, logger: Optional[logging.Logger] = None, config: EvalConfig = EvalConfig()):
        self.logger = logger or self._create_logger()
        self.config = config
        
    def _create_logger(self) -> logging.Logger:
        """Create logger"""
        logger = logging.getLogger(__name__)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        logger.setLevel(logging.DEBUG)
        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-5s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _save_json_results(self, result: Dict[str, Any], json_file: Path, 
                          base_dir: Path, cuda_file: Path, torch_ref_file: Path):
        """Save evaluation results to JSON file with detailed information"""
        try:
            # Prepare detailed JSON result
            # Handle both absolute and relative paths properly
            try:
                cuda_rel_path = str(cuda_file.relative_to(base_dir))
            except ValueError:
                # If relative_to fails, just use the filename
                cuda_rel_path = cuda_file.name
            
            try:
                torch_rel_path = str(torch_ref_file.relative_to(base_dir))
            except ValueError:
                # If relative_to fails, just use the filename
                torch_rel_path = torch_ref_file.name
            
            json_result = {
                "timestamp": datetime.now().isoformat(),
                "test_case": base_dir.name,
                "files": {
                    "cuda": cuda_rel_path,
                    "torch": torch_rel_path
                },
                "success": result.get("success", False)
            }
            
            # Always save test data if available, regardless of success status
            if result.get("correctness_details") or result.get("performance_details"):
                # Add detailed correctness results
                if "correctness_details" in result:
                    cuda_torch_compare = result["correctness_details"]["cuda_torch_compare"]
                    
                    # Extract actual attributes from CompareResult object
                    if hasattr(cuda_torch_compare, '__dict__'):
                        # It's an object, extract attributes
                        json_result["correctness"] = {
                            "comp_exec_success": getattr(cuda_torch_compare, 'comp_exec_success', True),
                            "dtype_match": getattr(cuda_torch_compare, 'dtype_match', True),
                            "shape_match": getattr(cuda_torch_compare, 'shape_match', True),
                            "values_match": getattr(cuda_torch_compare, 'values_match', cuda_torch_compare.overall_match),
                            "overall_match": cuda_torch_compare.overall_match,
                            "max_relative_error": cuda_torch_compare.max_relative_error,
                            "max_absolute_error": cuda_torch_compare.max_absolute_error,
                            "element_match_percentage": getattr(cuda_torch_compare, 'element_match_percentage', None),
                            "error": getattr(cuda_torch_compare, 'error', ''),
                            "traceback": getattr(cuda_torch_compare, 'traceback', ''),
                            "output_capture": getattr(cuda_torch_compare, 'output_capture', '')
                        }
                    else:
                        # It's already a dict
                        json_result["correctness"] = {
                            "comp_exec_success": cuda_torch_compare.get('comp_exec_success', True),
                            "dtype_match": cuda_torch_compare.get('dtype_match', True),
                            "shape_match": cuda_torch_compare.get('shape_match', True),
                            "values_match": cuda_torch_compare.get('values_match', cuda_torch_compare.get("overall_match", False)),
                            "overall_match": cuda_torch_compare.get("overall_match", False),
                            "max_relative_error": cuda_torch_compare.get("max_relative_error", 0.0),
                            "max_absolute_error": cuda_torch_compare.get("max_absolute_error", 0.0),
                            "element_match_percentage": cuda_torch_compare.get("element_match_percentage"),
                            "error": cuda_torch_compare.get('error', ''),
                            "traceback": cuda_torch_compare.get('traceback', ''),
                            "output_capture": cuda_torch_compare.get('output_capture', '')
                        }
                else:
                    # Fallback for basic correctness info
                    correctness = result.get("correctness", {})
                    json_result["correctness"] = {
                        "comp_exec_success": True,
                        "dtype_match": True,
                        "shape_match": True,
                        "values_match": correctness.get("cuda_torch_match", False),
                        "overall_match": correctness.get("cuda_torch_match", False),
                        "max_relative_error": 0.0,
                        "max_absolute_error": 0.0,
                        "element_match_percentage": None,
                        "error": '',
                        "traceback": '',
                        "output_capture": ''
                    }
                
                # Add detailed performance results
                if "performance_details" in result:
                    perf_details = result["performance_details"]
                    
                    # CUDA performance
                    cuda_perf = perf_details.get("cuda_perf")
                    cuda_time_ms = 0.0
                    if cuda_perf:
                        if hasattr(cuda_perf, 'perf_time_ms'):
                            cuda_time_ms = cuda_perf.perf_time_ms
                        elif isinstance(cuda_perf, dict):
                            cuda_time_ms = cuda_perf.get("perf_time_ms", 0.0)
                    
                    # PyTorch performance
                    torch_perf = perf_details.get("torch_perf")
                    torch_time_ms = 0.0
                    if torch_perf:
                        if hasattr(torch_perf, 'perf_time_ms'):
                            torch_time_ms = torch_perf.perf_time_ms
                        elif isinstance(torch_perf, dict):
                            torch_time_ms = torch_perf.get("perf_time_ms", 0.0)
                    
                    # Calculate speedup
                    cuda_speedup = 1.0
                    pytorch_speedup = cuda_time_ms / torch_time_ms if torch_time_ms > 0 else 0.0
                    
                    json_result["performance"] = {
                        "cuda": {
                            "perf_exec_success": getattr(cuda_perf, 'perf_exec_success', True) if cuda_perf else False,
                            "perf_time_ms": round(cuda_time_ms, 5),
                            "error": getattr(cuda_perf, 'error', '') if cuda_perf else 'Performance test not executed',
                            "traceback": getattr(cuda_perf, 'traceback', '') if cuda_perf else '',
                            "output_capture": getattr(cuda_perf, 'output_capture', '') if cuda_perf else '',
                            "warmup_runs": getattr(cuda_perf, 'warmup_runs', None) if cuda_perf else None,
                            "test_runs": getattr(cuda_perf, 'test_runs', None) if cuda_perf else None
                        },
                        "torch": {
                            "perf_exec_success": getattr(torch_perf, 'perf_exec_success', True) if torch_perf else False,
                            "perf_time_ms": round(torch_time_ms, 5),
                            "error": getattr(torch_perf, 'error', '') if torch_perf else 'Performance test not executed',
                            "traceback": getattr(torch_perf, 'traceback', '') if torch_perf else '',
                            "output_capture": getattr(torch_perf, 'output_capture', '') if torch_perf else '',
                            "warmup_runs": getattr(torch_perf, 'warmup_runs', None) if torch_perf else None,
                            "test_runs": getattr(torch_perf, 'test_runs', None) if torch_perf else None
                        },
                        "summary": {
                            "cuda_time_ms": round(cuda_time_ms, 5),
                            "torch_time_ms": round(torch_time_ms, 5),
                            "cuda_speedup": round(cuda_speedup, 2),
                            "pytorch_speedup": round(pytorch_speedup, 2)
                        }
                    }
                else:
                    # Fallback for basic performance info
                    perf = result.get("performance", {})
                    cuda_time = perf.get("cuda_time", 0.0)
                    torch_time = perf.get("torch_time", 0.0)
                    
                    json_result["performance"] = {
                        "cuda": {
                            "perf_exec_success": True,
                            "perf_time_ms": round(cuda_time, 5),
                            "error": '',
                            "traceback": '',
                            "output_capture": ''
                        },
                        "torch": {
                            "perf_exec_success": True,
                            "perf_time_ms": round(torch_time, 5),
                            "error": '',
                            "traceback": '',
                            "output_capture": ''
                        },
                        "summary": {
                            "cuda_time_ms": round(cuda_time, 5),
                            "torch_time_ms": round(torch_time, 5),
                            "cuda_speedup": 1.0,
                            "pytorch_speedup": round(perf.get("cuda_torch_speedup", 0.0), 2)
                        }
                    }
                
                # Add device info
                device_info = result.get("device_info", {})
                json_result["device_info"] = {
                    "device_name": device_info.get("device_name", "Unknown"),
                    "device_id": device_info.get("device_id", 0)
                }
                
            else:
                # No test data available - use default empty structures
                json_result["correctness"] = {
                    "comp_exec_success": False,
                    "dtype_match": False,
                    "shape_match": False,
                    "values_match": False,
                    "overall_match": False,
                    "max_relative_error": 0.0,
                    "max_absolute_error": 0.0,
                    "element_match_percentage": None,
                    "error": "No test data available",
                    "traceback": "",
                    "output_capture": ''
                }
                
                json_result["performance"] = {
                    "cuda": {
                        "perf_exec_success": False,
                        "perf_time_ms": 0.0,
                        "error": "No test data available",
                        "traceback": "",
                        "output_capture": ''
                    },
                    "torch": {
                        "perf_exec_success": False,
                        "perf_time_ms": 0.0,
                        "error": "No test data available",
                        "traceback": "",
                        "output_capture": ''
                    },
                    "summary": {
                        "cuda_time_ms": 0.0,
                        "torch_time_ms": 0.0,
                        "cuda_speedup": 0.0,
                        "pytorch_speedup": 0.0
                    }
                }
            
            # Add error information if the test failed  
            if not result.get("success"):
                error = result.get("error", {})
                if isinstance(error, dict):
                    json_result["error"] = {
                        "message": error.get("message", "Test failed - check details above"),
                        "type": error.get("type", "TestFailure"),
                        "traceback": error.get("traceback", ""),
                        "traceback_lines": error.get("traceback", "").split('\n') if error.get("traceback") else []
                    }
                else:
                    json_result["error"] = {
                        "message": str(error) if error else "Test failed - check details above",
                        "type": "TestFailure",
                        "traceback": "",
                        "traceback_lines": []
                    }
            
            # Write JSON file
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📄 Detailed results saved to: {json_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save JSON results: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _setup_file_logging(self, base_dir: Path, model_name: str, time_str: str, 
                           logfile_prefix: str = "", timestamp_log_dir: Optional[Path] = None) -> Path:
        """Setup file logging"""
        # Use fixed directory name "check_cuda" instead of timestamp
        log_dir = base_dir / "logs" / "check_cuda"
        os.makedirs(log_dir, exist_ok=True)
        log_file = log_dir / "eval.log"
        
        # Check if file handler already exists for this file
        file_handler_exists = False
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file.resolve()):
                file_handler_exists = True
                break
        
        # Only add file handler if it doesn't exist
        if not file_handler_exists:
            file_handler = logging.FileHandler(log_file, mode='w')  # Overwrite mode
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-5s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        return log_dir
    
    def _run_compare_test(self, worker_func: Callable, test_name: str, 
                         config: EvalConfig, output_capture_file: Optional[str], 
                         timeout: int = DEFAULT_CONFIG().subproc_timeout, **kwargs) -> Tuple[bool, CompareResult, str]:
        """Run comparison test with generic function"""
        self.logger.info(f"🔍 {test_name}...")
        
        # Convert any Path objects to strings and ensure absolute paths
        serializable_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(value, '__fspath__'):  # Path-like object
                # Convert to absolute path to ensure GPU server can find files
                serializable_kwargs[key] = str(Path(value).resolve())
            else:
                serializable_kwargs[key] = value
        
        # Build args list for CUDA vs PyTorch comparison
        args_list = [serializable_kwargs.get("cuda_file"), serializable_kwargs.get("torch_ref_file"), DEFAULT_RANDOM_SEED, config.to_dict(), output_capture_file, timeout]
        
        from server.xpu.nvgpu.task_queue import TaskType
        task_data = {
            "task_type": TaskType.FUNCTIONAL.value,
            "name": test_name,
            "description": f"Comparison test: {test_name}",
            "module_path": "eval_.kernelbench_c.custom_mprunner",
            "function_name": worker_func.__name__,
            "args": args_list,
            "kwargs": {},
            "preferred_gpu_id": 0,
            "allow_fallback": False,
            "require_same_gpu_type": True
        }
        task_id = submit_task(task_data)
        self.logger.info(f"✅ Submitted task: {task_id}")
        
        status = wait_for_task_completion(task_id, timeout=config.subproc_timeout + 5)
        self.logger.info(f"✅ Task completed: {status}")
        
        result = get_task_status(task_id)
        self.logger.info(f"✅ Task result: {result}")
        
        # Handle API result format
        if result.get("status") == "completed" and result.get("result"):
            task_result = result["result"]
            if hasattr(task_result, 'subproc_success') and task_result.subproc_success:
                compare_result = task_result.result
                self.logger.info(
                    f"    ✓ {test_name}: "
                    f"max_rel_err={compare_result.max_relative_error:.3f}, "
                    f"max_abs_err={compare_result.max_absolute_error:.6f}"
                )
                return True, compare_result, ""
            elif isinstance(task_result, dict) and task_result.get("subproc_success"):
                compare_result_dict = task_result["result"]
                from eval_.common.verifier import CompareResult
                compare_result = CompareResult(**compare_result_dict)
                self.logger.info(
                    f"    ✓ {test_name}: "
                    f"max_rel_err={compare_result.max_relative_error:.3f}, "
                    f"max_abs_err={compare_result.max_absolute_error:.6f}"
                )
                return True, compare_result, ""
            else:
                error_msg = f"{test_name} failed - {getattr(task_result, 'error', task_result.get('error', 'Unknown error'))}"
                self.logger.error(f"    ❌ {error_msg}")
                return False, None, error_msg
        else:
            error_msg = f"{test_name} failed - {result.get('error', 'Task failed')}"
            self.logger.error(f"    ❌ {error_msg}")
            return False, None, error_msg
    
    def _run_perf_test(self, worker_func: Callable, test_name: str,
                      config: EvalConfig, output_capture_file: Optional[str],
                      **kwargs) -> Tuple[bool, PerformanceResult, str]:
        """Run performance test with generic function"""
        self.logger.info(f"🚀 {test_name}...")
        
        # Convert any Path objects to strings and ensure absolute paths
        serializable_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(value, '__fspath__'):  # Path-like object
                # Convert to absolute path to ensure GPU server can find files
                serializable_kwargs[key] = str(Path(value).resolve())
            else:
                serializable_kwargs[key] = value
        
        # Build args list based on function signature
        args_list = []
        if worker_func.__name__ == "cuda_perf_worker":
            args_list = [serializable_kwargs.get("cuda_file"), serializable_kwargs.get("torch_ref_file"), DEFAULT_RANDOM_SEED, output_capture_file, config.to_dict()]
        elif worker_func.__name__ == "torch_perf_worker":
            args_list = [serializable_kwargs.get("torch_file"), DEFAULT_RANDOM_SEED, output_capture_file, config.to_dict()]
        else:
            args_list = [config.to_dict(), output_capture_file]
            
        from server.xpu.nvgpu.task_queue import TaskType
        task_data = {
            "task_type": TaskType.PERFORMANCE.value,
            "name": test_name,
            "description": f"Performance test: {test_name}",
            "module_path": "eval_.kernelbench_c.custom_mprunner",
            "function_name": worker_func.__name__,
            "args": args_list,
            "kwargs": {},
            "preferred_gpu_id": 1,
            "allow_fallback": False,
            "require_same_gpu_type": True
        }
        task_id = submit_task(task_data)
        self.logger.info(f"✅ Submitted task: {task_id}")
        
        status = wait_for_task_completion(task_id, timeout=config.subproc_timeout + 5)
        self.logger.info(f"✅ Task completed: {status}")
        
        result = get_task_status(task_id)
        self.logger.info(f"✅ Task result: {result}")
        
        # Handle API result format
        if result.get("status") == "completed" and result.get("result"):
            task_result = result["result"]
            if hasattr(task_result, 'subproc_success') and task_result.subproc_success:
                perf_result = task_result.result
                self.logger.info(f"    ✓ {test_name}: {perf_result.perf_time_ms:.3f} ms")
                return True, perf_result, ""
            elif isinstance(task_result, dict) and task_result.get("subproc_success"):
                perf_result_dict = task_result["result"]
                from eval_.common.verifier import PerformanceResult
                perf_result = PerformanceResult(**perf_result_dict)
                self.logger.info(f"    ✓ {test_name}: {perf_result.perf_time_ms:.3f} ms")
                return True, perf_result, ""
            else:
                error_msg = f"{test_name} failed - {getattr(task_result, 'error', task_result.get('error', 'Unknown error'))}"
                self.logger.error(f"    ❌ {error_msg}")
                return False, None, error_msg
        else:
            error_msg = f"{test_name} failed - {result.get('error', 'Task failed')}"
            self.logger.error(f"    ❌ {error_msg}")
            return False, None, error_msg
    
    def _run_evaluation_core(self, cuda_file: str, torch_ref_file: str,
                           config: EvalConfig, output_capture_file: Optional[str], timeout: int = DEFAULT_CONFIG().subproc_timeout) -> Dict[str, Any]:
        """Core evaluation logic - Only CUDA vs PyTorch"""
        try:
            # ═══════════════ Correctness Testing ═══════════════
            self.logger.info("🧪 Correctness Testing")
            self.logger.info("─" * 50)
            
            test_name = "Correctness" if config.cuda_kernel_name == "cuda_kernel" else f"[{config.cuda_kernel_name}] - Correctness"
            
            # CUDA vs PyTorch
            success, cuda_torch_compare, error = self._run_compare_test(
                cuda_compare_torch_worker, test_name + " - CUDA vs PyTorch",
                config, output_capture_file, timeout,
                cuda_file=cuda_file, torch_ref_file=torch_ref_file
            )
            if not success:
                return {"success": False, "error": error, "subprocess_failed": True}
            
            # Output correctness summary
            self.logger.info("─" * 50)
            self.logger.info("📋 Correctness Summary:")
            self.logger.info(f"    CUDA vs PyTorch:   {'✅ PASS' if cuda_torch_compare.overall_match else '❌ FAIL'}")
            
            correctness = {
                "cuda_torch_match": cuda_torch_compare.overall_match
            }
            
            # Always save detailed correctness info
            correctness_details = {
                "cuda_torch_compare": cuda_torch_compare
            }
            
            # Continue with performance testing even if correctness fails
            # (Performance data is still useful for analysis)
            
            # ═══════════════ Performance Testing ═══════════════
            self.logger.info("")
            self.logger.info("⚡ Performance Testing")
            self.logger.info("─" * 50)
            
            test_name = "Perf" if config.cuda_kernel_name == "cuda_kernel" else f"[{config.cuda_kernel_name}] - Perf"
            
            # CUDA performance test
            cuda_success, cuda_perf, cuda_error = self._run_perf_test(
                cuda_perf_worker, test_name + " - CUDA",
                config, output_capture_file,
                cuda_file=cuda_file, torch_ref_file=torch_ref_file
            )
            
            # PyTorch performance test
            torch_success, torch_perf, torch_error = self._run_perf_test(
                torch_perf_worker, test_name + " - PyTorch",
                config, output_capture_file,
                torch_file=torch_ref_file
            )
            
            # Even if performance tests fail, we want to save what we have
            
            # Performance summary
            cuda_time = cuda_perf.perf_time_ms if cuda_success and cuda_perf else 0.0
            torch_time = torch_perf.perf_time_ms if torch_success and torch_perf else 0.0
            
            cuda_torch_speedup = torch_time / cuda_time if cuda_time > 0 else 0.0
            
            if cuda_success and torch_success:
                self.logger.info("─" * 50)
                self.logger.info("📈 Performance Test Results:")
                self.logger.info(f"  CUDA: {cuda_time:.3f} ms")
                self.logger.info(f"  PyTorch: {torch_time:.3f} ms")
                self.logger.info("─" * 30)
                self.logger.info(f"    CUDA vs PyTorch speedup: {cuda_torch_speedup:6.2f}x")
            
            # Get device info
            device_name = "Unknown"
            device_id = 0
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(device_id)
            
            # Determine overall success: correctness must pass, performance is optional
            overall_success = cuda_torch_compare.overall_match
            
            return {
                "success": overall_success,
                "performance": {
                    "cuda_time": cuda_time,
                    "torch_time": torch_time,
                    "cuda_torch_speedup": cuda_torch_speedup
                },
                "correctness": correctness,
                "correctness_details": correctness_details,
                "performance_details": {
                    "cuda_perf": cuda_perf if cuda_success else None,
                    "torch_perf": torch_perf if torch_success else None
                },
                "device_info": {"device_name": device_name, "device_id": device_id},
                "output_capture": (
                    getattr(cuda_torch_compare, 'output_capture', '') +
                    (getattr(cuda_perf, 'output_capture', '') if cuda_success else '') +
                    (getattr(torch_perf, 'output_capture', '') if torch_success else '')
                ),
                "errors": {
                    "correctness_error": "" if cuda_torch_compare.overall_match else "Correctness test failed: result mismatch",
                    "cuda_perf_error": cuda_error if not cuda_success else "",
                    "torch_perf_error": torch_error if not torch_success else ""
                }
            }
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            self.logger.error(f"❌ Evaluation error: {error_msg}")
            self.logger.error(f"Traceback:\n{error_traceback}")
            return {
                "success": False, 
                "error": {
                    "message": error_msg,
                    "traceback": error_traceback,
                    "type": type(e).__name__
                }
            }

    async def evaluate_cuda_kernel(self, base_dir: Path, kernel_name: str = None,
                                 model_name: str = "cuda_test", time_str: Optional[str] = None,
                                 logfile_prefix: str = "", timestamp_log_dir: Optional[Path] = None,
                                 return_result: bool = False, timeout: int = 300,
                                 capture_output: bool = True) -> Optional[Dict[str, Any]]:
        """Evaluate CUDA kernel vs PyTorch"""
        
        if time_str is None:
            time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Determine file paths
        cuda_file = base_dir / "cuda_ref.cu"
        torch_ref_file = base_dir / "torch_ref.py"
        
        cuda_file = cuda_file.resolve()
        torch_ref_file = torch_ref_file.resolve()
        
        # Setup logging
        log_dir = self._setup_file_logging(base_dir, model_name, time_str, logfile_prefix, timestamp_log_dir)
        
        log_dir = log_dir.resolve()
        json_result_file = log_dir / "results.json"
        
        self.logger.info("")
        self.logger.info("╔" + "═" * 58 + "╗")
        self.logger.info("║" + f"{'CUDA Kernel Evaluation Started':^58}" + "║")
        self.logger.info("║" + f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^58}" + "║")
        self.logger.info("╚" + "═" * 58 + "╝")
        
        try:
            # Check file existence
            for file_path, desc in [(cuda_file, "CUDA"), (torch_ref_file, "PyTorch")]:
                if not os.path.exists(file_path):
                    error_msg = f"{desc} file not found: {file_path}"
                    self.logger.error(f"❌ {error_msg}")
                    return {"success": False, "error": error_msg} if return_result else None
            
            kernel_name = Path(base_dir).name
            build_dir = Path(base_dir).resolve() / "build"
            self.logger.info(f"build_dir: {build_dir}")
            
            # Prepare configuration
            config = EvalConfig(
                cuda_kernel_name=kernel_name,
                build_dir=str(build_dir),
                atol=self.config.atol,
                rtol=self.config.rtol,
                subproc_timeout=timeout
            )
            
            output_capture_file = str(log_dir / f"subprocess_output_{time_str}.log") if capture_output else None
            
            # Ensure output capture file directory exists
            if output_capture_file:
                os.makedirs(os.path.dirname(output_capture_file), exist_ok=True)
            
            self.logger.info("📁 File Paths:")
            self.logger.info(f"    CUDA:     {cuda_file}")
            self.logger.info(f"    PyTorch:  {torch_ref_file}")
            self.logger.info("")
            
            # Execute evaluation
            result = await asyncio.to_thread(
                self._run_evaluation_core, str(cuda_file), str(torch_ref_file), config, output_capture_file, timeout
            )
            
            # Save JSON results
            self._save_json_results(result, json_result_file, base_dir, cuda_file, torch_ref_file)
            
            # Log results
            self.logger.info("")
            if result.get("success"):
                self.logger.info("╔" + "═" * 58 + "╗")
                self.logger.info("║" + f"{'🎉 Evaluation Completed Successfully!':^57}" + "║")
                self.logger.info("╚" + "═" * 58 + "╝")
            else:
                self.logger.error("╔" + "═" * 58 + "╗")
                self.logger.error("║" + f"{'❌ Evaluation Failed':^57}" + "║")
                error_msg = result.get('error', 'Unknown error')
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get('message', 'Unknown error')
                self.logger.error("║" + f"{str(error_msg)[:54]:^58}" + "║")
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
async def test_cuda_vs_pytorch():
    """Test CUDA vs PyTorch performance"""
    config = EvalConfig(atol=0.001, rtol=0.001)
    evaluator = CUDAKernelEvaluator(config=config)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_dir = Path("outputs") / "tests" / "31_ELU"
    print(test_dir.resolve())
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    print(f"🚀 Testing directory: {test_dir}")
    result = await evaluator.evaluate_cuda_kernel(
        base_dir=test_dir,
        model_name="cuda_pytorch_comparison",
        return_result=True,
        timeout=180
    )
    
    if result:
        print(f"\n📊 Final result: {'Success' if result['success'] else 'Failed'}")
        if result.get('performance'):
            perf = result['performance']
            print(f"Performance: CUDA {perf['cuda_time']:.3f}ms, PyTorch {perf['torch_time']:.3f}ms, Speedup {perf['cuda_torch_speedup']:.2f}x")
        if not result['success']:
            print(f"❌ Error: {result.get('error')}")


async def main():
    """Main function"""
    print("🎯 CUDA Kernel Evaluator - CUDA vs PyTorch Only")
    print("=" * 50)
    await test_cuda_vs_pytorch()


if __name__ == "__main__":
    asyncio.run(main())
