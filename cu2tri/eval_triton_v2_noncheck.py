# -*- coding: utf-8 -*-
"""
Triton内核评估模块

该模块提供了对Triton内核进行编译、测试和性能评估的功能，
支持与CUDA和PyTorch参考实现的对比分析。
"""

import os
import sys
import asyncio
import logging
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import gc

# 设置CUDA环境变量
os.environ["TORCH_USE_CUDA_DSA"] = "1" 
os.environ['TORCH_CUDA_ARCH_LIST'] = "Ada"

# 导入项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# 添加项目根目录到Python路径
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import torch
    from eval_.kernelbench_c.custom_loader import load_torch_ref_from_pyfile, load_triton_kernel_from_pyfile
    from eval_.common.loader import load_cuda_extension_from_cufile
    from eval_.common.config import EvalConfig
    from eval_.common.benchmark import benchmark_kernel
    from eval_.common.mprunner import mp_run
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure the necessary dependencies are installed")
    sys.exit(1)


class TritonKernelEvaluator:
    """Triton内核评估器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None, config: EvalConfig = EvalConfig()):
        """初始化评估器
        
        Args:
            logger: 日志记录器，如果未提供则创建默认记录器
        """
        self.logger = logger or self._create_default_logger()
        self.config = config
        
    def _create_default_logger(self) -> logging.Logger:
        """创建默认日志记录器"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    
    def benchmark_kernel(self, kernel_func: callable, inputs: List[torch.Tensor]) -> float:
        """对内核进行性能基准测试
        
        Args:
            kernel_func: 内核函数
            inputs: 输入张量列表
            
        Returns:
            平均执行时间（毫秒）
        """
        try:
            return benchmark_kernel(kernel_func, inputs)
        except Exception as e:
            self.logger.error(f"Performance Test Failed: {e}")
            return float('inf')
    
    def check_cuda_availability(self) -> bool:
        """检查CUDA是否可用"""
        if not torch.cuda.is_available():
            self.logger.error("❌ CUDA Unavailable, Skip Test")
            return False
        return True
    
    def prepare_test_inputs(self, torch_ref_module: Any) -> List[torch.Tensor]:
        """准备测试输入数据
        
        Args:
            torch_ref_module: PyTorch参考模块
            
        Returns:
            CUDA上的测试输入列表
        """
        test_inputs = torch_ref_module.get_inputs()
        cuda_inputs = [
            inp.cuda() if isinstance(inp, torch.Tensor) else inp 
            for inp in test_inputs
        ]
        
        if cuda_inputs:
            self.logger.info(f"📊 Test Matrix Size: {cuda_inputs[0].shape}")
            
        return cuda_inputs
    
    def compare_results(
        self, 
        triton_result: torch.Tensor,
        cuda_result: torch.Tensor, 
        torch_result: torch.Tensor
    ) -> Dict[str, bool]:
        atol, rtol = self.config.atol, self.config.rtol
        
        comparisons = {
            "triton_cuda_match": torch.allclose(triton_result, cuda_result, atol=atol, rtol=rtol),
            "triton_torch_match": torch.allclose(triton_result, torch_result, atol=atol, rtol=rtol),
            "cuda_torch_match": torch.allclose(cuda_result, torch_result, atol=atol, rtol=rtol)
        }
        
        self.logger.info(f"🔍 Triton vs CUDA Match: {'✅' if comparisons['triton_cuda_match'] else '❌'}")
        self.logger.info(f"🔍 Triton vs PyTorch Match: {'✅' if comparisons['triton_torch_match'] else '❌'}")
        self.logger.info(f"🔍 CUDA vs PyTorch Match: {'✅' if comparisons['cuda_torch_match'] else '❌'}")
        
        if not comparisons["triton_cuda_match"]:
            max_diff = torch.max(torch.abs(triton_result - cuda_result)).item()
            self.logger.info(f"   Max Difference: {max_diff:.2e}")
            
        return comparisons
    
    def setup_logging(
        self, 
        base_dir: str, 
        model_name: str, 
        time_str: str,
        logfile_prefix: str = "",
        timestamp_log_dir: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """设置日志记录
        
        Args:
            base_dir: 基础目录
            model_name: 模型名称
            time_str: 时间戳字符串
            logfile_prefix: 日志文件前缀
            timestamp_log_dir: 时间戳日志目录
            
        Returns:
            (主日志文件路径, 最新日志文件路径)
        """
        log_dir = f"{base_dir}/logs"
        model_name_clean = model_name.replace('.', '_').replace('/', '_').replace('-', '_')
        model_log_dir = f"{log_dir}/{model_name_clean}"
        
        # 确定日志文件路径
        if timestamp_log_dir is not None:
            log_file = f"{timestamp_log_dir}/{logfile_prefix}eval.log"
            latest_eval_file = None if logfile_prefix.startswith("round") else f"{model_log_dir}/eval_latest.log"
        else:
            os.makedirs(model_log_dir, exist_ok=True)
            log_file = f"{model_log_dir}/{logfile_prefix}eval_{time_str}.log"
            latest_eval_file = None
        
        # 确保目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        if latest_eval_file:
            os.makedirs(os.path.dirname(latest_eval_file), exist_ok=True)
        
        return log_file, latest_eval_file
    
    def configure_file_logging(self, log_file: str, latest_eval_file: Optional[str] = None) -> None:
        """配置文件日志记录
        
        Args:
            log_file: 主日志文件路径
            latest_eval_file: 最新日志文件路径（可选）
        """
        # 清除之前的文件处理器
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)
        
        # 添加主日志处理器
        main_handler = logging.FileHandler(log_file, mode='a')
        main_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(main_handler)
        
        # 如果需要，添加最新日志副本处理器
        if latest_eval_file:
            latest_handler = logging.FileHandler(latest_eval_file, mode='w')
            latest_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(latest_handler)
        
        self.logger.setLevel(logging.DEBUG)
    
    def log_test_start(self, logfile_prefix: str = "") -> None:
        """记录测试开始信息"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        round_info = f" - {logfile_prefix.replace('_', '').upper()}" if logfile_prefix.startswith("round") else ""
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Test Start Time: {current_time}{round_info}")
        self.logger.info(f"{'='*60}")
    
    def log_test_end(self) -> None:
        """记录测试结束信息"""
        self.logger.info("\n✅ Test Completed!")
        self.logger.info(f"Test End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'='*60}\n")
    
    def cleanup_gpu_memory(self) -> None:
        """清理GPU内存"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            self.logger.warning(f"Clean GPU Memory Error: {e}")

    async def evaluate_triton_kernel(
        self,
        base_dir: str,
        kernel_name: str = None,
        model_name: str = "triton_test",
        time_str: Optional[str] = None,
        logfile_prefix: str = "",
        timestamp_log_dir: Optional[str] = None,
        return_result: bool = False,
        timeout: int = 300
    ) -> Optional[Dict[str, Any]]:
        """评估Triton内核（使用子进程模式避免core dump导致主进程崩溃）
        
        Args:
            base_dir: 基础目录路径
            kernel_name: CUDA内核名称
            model_name: 模型名称
            time_str: 时间戳字符串
            logfile_prefix: 日志文件前缀
            timestamp_log_dir: 时间戳日志目录
            return_result: 是否返回结果字典
            timeout: 子进程超时时间（秒）
            
        Returns:
            如果return_result为True，返回包含评估结果的字典
        """
        # 确定时间戳
        if time_str is None:
            time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 动态确定triton文件路径
        if timestamp_log_dir and logfile_prefix.startswith("round"):
            round_num = logfile_prefix.replace("round", "").replace("_", "")
            triton_file = f"{timestamp_log_dir}/triton_round{round_num}.py"
        elif timestamp_log_dir:
            triton_file = f"{timestamp_log_dir}/triton.py"
        else:
            triton_file = f"{base_dir}/triton.py"
        
        # 其他文件路径
        cuda_file = f"{base_dir}/cuda_ref.cu"
        torch_ref_file = f"{base_dir}/torch_ref.py"
        
        # 设置日志
        log_file, latest_eval_file = self.setup_logging(
            base_dir, model_name, time_str, logfile_prefix, timestamp_log_dir
        )
        self.configure_file_logging(log_file, latest_eval_file)
        
        self.log_test_start(logfile_prefix)
        
        try:
            # 检查必要文件是否存在
            for file_path, file_desc in [
                (triton_file, "Triton kernel file"),
                (cuda_file, "CUDA reference file"),
                (torch_ref_file, "PyTorch reference file")
            ]:
                if not os.path.exists(file_path):
                    error_msg = f"{file_desc} does not exist: {file_path}"
                    self.logger.error(f"❌ {error_msg}")
                    result = {"success": False, "error": error_msg}
                    return result if return_result else None
            
            # 准备配置字典
            config_dict = {
                'atol': self.config.atol,
                'rtol': self.config.rtol
            }
            
            self.logger.info(f"🚀 Start subprocess to evaluate Triton kernel (timeout: {timeout} seconds)")
            
            # 使用子进程执行评估，避免core dump影响主进程
            subprocess_result = await asyncio.to_thread(
                mp_run,
                worker_func=evaluate_triton_kernel_subproc,
                args=(
                    base_dir,
                    kernel_name,
                    triton_file,
                    cuda_file,
                    torch_ref_file,
                    config_dict
                ),
                timeout=timeout
            )
            
            # 处理子进程结果
            if subprocess_result.subproc_success:
                result = subprocess_result.result
                self.logger.info("✅ Subprocess Evaluation Completed Successfully")
                
                # 记录详细结果到主进程日志
                if result.get("success"):
                    if "performance" in result:
                        perf = result["performance"]
                        self.logger.info(f"\n📈 Performance Test Results:")
                        self.logger.info(f"   Triton:   {perf['triton_time']:.3f} ms")
                        self.logger.info(f"   CUDA:     {perf['cuda_time']:.3f} ms")
                        self.logger.info(f"   PyTorch:  {perf['torch_time']:.3f} ms")
                        self.logger.info(f"\n🚀 Speedup:")
                        self.logger.info(f"   Triton vs CUDA: {perf['triton_cuda_speedup']:.2f}x")
                        self.logger.info(f"   Triton vs PyTorch: {perf['triton_torch_speedup']:.2f}x")
                    
                    if "device_info" in result:
                        device_info = result["device_info"]
                        self.logger.info(f"📱 GPU Device Information: {device_info['device_name']} (Device {device_info['device_id']})")
                else:
                    self.logger.info(f"❌ Subprocess Evaluation Failed: {result.get('error', 'Unknown error')}")
                
                self.log_test_end()
                return result if return_result else None
            else:            
                error_msg = f"Subprocess Execution Failed: {subprocess_result.error}"
                self.logger.error(f"❌ {error_msg}")
                result = {
                    "success": False,
                    "error": error_msg,
                    "subprocess_failed": True
                }
                return result if return_result else None
            
        except Exception as e:
            self.logger.error(f"❌ Main Process Evaluation Error: {e}")
            import traceback
            result = {
                "success": False, 
                "error": str(e), 
                "traceback": traceback.format_exc(),
                "main_process_error": True
            }
            return result if return_result else None
        
        finally:
            # 主进程也进行清理
            self.cleanup_gpu_memory()


def evaluate_triton_kernel_subproc(
    result_queue: multiprocessing.Queue,
    base_dir: str,
    kernel_name: Optional[str],
    triton_file: str,
    cuda_file: str,
    torch_ref_file: str,
    config_dict: Dict[str, Any]
) -> None:
    """在子进程中执行Triton内核评估的核心逻辑
    
    Args:
        result_queue: 用于返回结果的队列
        base_dir: 基础目录路径
        kernel_name: CUDA内核名称
        triton_file: Triton文件路径
        cuda_file: CUDA文件路径
        torch_ref_file: PyTorch参考文件路径
        config_dict: 配置字典
    """
    import multiprocessing
    import torch
    import gc
    import logging
    
    # 在子进程中设置独立的日志系统
    process_id = multiprocessing.current_process().pid
    logger = logging.getLogger(f"TritonEval-Subprocess-{process_id}")
    logger.setLevel(logging.INFO)
    
    # 清除可能存在的处理器，避免重复
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 设置控制台输出
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        f'[PID-{process_id}] %(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 设置子进程专用的日志文件（避免与主进程竞争）
    try:
        subprocess_log_file = f"{base_dir}/logs/subprocess_{process_id}.log"
        import os
        os.makedirs(os.path.dirname(subprocess_log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(subprocess_log_file, mode='w')
        file_formatter = logging.Formatter(
            '[%(levelname)s]\t%(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as log_setup_error:
        # 如果日志文件设置失败，不影响主要功能
        logger.warning(f"Failed to setup subprocess log file: {log_setup_error}")
    
    # 防止日志向上传播，避免与主进程日志混淆
    logger.propagate = False
    
    result = {}
    result_queue_filled = False  # 初始化结果队列填充标记
    
    try:
        logger.info(f"Subprocess {process_id} start Triton kernel evaluation")
        
        # 检查CUDA可用性
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        logger.info(f"Use GPU device: {device_name} (Device {current_device})")
        
        # 创建配置对象
        config = EvalConfig(
            cuda_kernel_name=kernel_name or "cuda_kernel",
            build_dir=f"{base_dir}/build",
            atol=config_dict.get('atol', 1e-1),
            rtol=config_dict.get('rtol', 1e-1)
        )
        
        # 加载测试数据
        logger.info("📁 Load test case...")
        torch_ref = load_torch_ref_from_pyfile(torch_ref_file)
        test_inputs = torch_ref.get_inputs()
        cuda_inputs = [
            inp.cuda() if isinstance(inp, torch.Tensor) else inp 
            for inp in test_inputs
        ]
        
        if cuda_inputs:
            logger.info(f"📊 Test matrix size: {cuda_inputs[0].shape}")
        
        # 加载Triton内核
        logger.info(f"⚡ Load Triton kernel: {triton_file}")
        triton_func = load_triton_kernel_from_pyfile(triton_file)
        logger.info("✅ Triton kernel loaded successfully")
        
        # 编译CUDA内核
        logger.info("🔧 Compile CUDA kernel...")
        cuda_extension = load_cuda_extension_from_cufile(cuda_file, config)
        cuda_func = getattr(cuda_extension, 'forward')
        logger.info("✅ CUDA kernel compiled successfully")
        
        # 功能正确性测试
        logger.info("\n🧪 Start function correctness test...")
        
        # 使用torch.no_grad()防止显存持续占用
        with torch.no_grad():
            triton_result = triton_func(*cuda_inputs)
            cuda_result = cuda_func(*cuda_inputs)
            torch_result = torch_ref.module_fn(*cuda_inputs)
        
        # 比较结果
        atol, rtol = config.atol, config.rtol
        comparisons = {
            "triton_cuda_match": torch.allclose(triton_result, cuda_result, atol=atol, rtol=rtol),
            "triton_torch_match": torch.allclose(triton_result, torch_result, atol=atol, rtol=rtol),
            "cuda_torch_match": torch.allclose(cuda_result, torch_result, atol=atol, rtol=rtol)
        }
        
        logger.info(f"🔍 Triton vs CUDA match: {'✅' if comparisons['triton_cuda_match'] else '❌'}")
        logger.info(f"🔍 Triton vs PyTorch match: {'✅' if comparisons['triton_torch_match'] else '❌'}")
        logger.info(f"🔍 CUDA vs PyTorch match: {'✅' if comparisons['cuda_torch_match'] else '❌'}")
        
        if not comparisons["triton_cuda_match"]:
            max_diff = torch.max(torch.abs(triton_result - cuda_result)).item()
            logger.info(f"   Max difference: {max_diff:.2e}")
        
        if not (comparisons["triton_cuda_match"] and comparisons["cuda_torch_match"]):
            logger.info("❌ Function test failed, skipping performance test")
            result = {"success": False, "error": "Function test failed: result mismatch", "correctness": comparisons}
        else:
            # 性能测试
            logger.info("\n⏱️  Start performance test...")
            
            triton_time = benchmark_kernel(triton_func, cuda_inputs)
            logger.info("Test CUDA kernel performance...")
            cuda_time = benchmark_kernel(cuda_func, cuda_inputs)
            logger.info("Test PyTorch reference performance...")
            torch_time = benchmark_kernel(torch_ref.module_fn, cuda_inputs)
            
            # 性能结果
            logger.info(f"\n📈 Performance test results:")
            logger.info(f"   Triton:   {triton_time:.3f} ms")
            logger.info(f"   CUDA:     {cuda_time:.3f} ms")
            logger.info(f"   PyTorch:  {torch_time:.3f} ms")
            
            # 计算加速比
            triton_cuda_speedup = cuda_time / triton_time if triton_time > 0 else 0.0
            triton_torch_speedup = torch_time / triton_time if triton_time > 0 else 0.0
            
            logger.info(f"\n🚀 Speedup:")
            logger.info(f"   Triton vs CUDA: {triton_cuda_speedup:.2f}x")
            logger.info(f"   Triton vs PyTorch: {triton_torch_speedup:.2f}x")
            
            result = {
                "success": True,
                "performance": {
                    "triton_time": triton_time,
                    "cuda_time": cuda_time,
                    "torch_time": torch_time,
                    "triton_cuda_speedup": triton_cuda_speedup,
                    "triton_torch_speedup": triton_torch_speedup
                },
                "correctness": comparisons,
                "device_info": {
                    "device_name": device_name,
                    "device_id": current_device
                }
            }
        
        logger.info("✅ Subprocess evaluation completed")
        result_queue.put(result)
        result_queue_filled = True  # 标记结果已放入队列
        
    except Exception as e:
        error_msg = f"Subprocess evaluation failed: {e}"
        logger.error(error_msg)
        import traceback
        error_result = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "process_id": process_id
        }
        result_queue.put(error_result)
        result_queue_filled = True  # 标记结果已放入队列
        
    finally:
        # 彻底清理GPU显存和上下文
        try:
            # 显式删除可能的张量引用
            tensor_vars = []
            for var_name, var_value in list(locals().items()):
                try:
                    if hasattr(var_value, 'dtype') and hasattr(var_value, 'device'):  # 更安全的torch.Tensor检查
                        tensor_vars.append(var_name)
                except:
                    pass
            
            for var_name in tensor_vars:
                try:
                    del locals()[var_name]
                except:
                    pass
            
            # 清理PyTorch CUDA上下文
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"Subprocess {process_id} cleaned up")
        except Exception as cleanup_error:
            # 即使清理失败，也要记录日志，但不阻止结果返回
            try:
                logger.error(f"Subprocess cleanup failed: {cleanup_error}")
            except:
                # 如果连日志都无法写入，则静默处理
                pass
        
        # 确保结果总是被放入队列（除非已经放入）
        try:
            # 检查是否已经放入结果
            if 'result_queue_filled' not in locals() or not result_queue_filled:
                result_queue.put({
                    "success": False,
                    "error": "Subprocess completed without putting result",
                    "process_id": process_id
                })
        except Exception:
            # 队列操作失败也要静默处理
            pass



# 主函数示例
async def main():
    """主函数示例"""
    evaluator = TritonKernelEvaluator()
    
    # 示例：评估单个内核
    test_dir = "outputs/kernelbench_c/level1/1_Square_matrix_multiplication_"
    if os.path.exists(test_dir):
        result = await evaluator.evaluate_triton_kernel(
            base_dir=test_dir,
            model_name="test_evaluation",
            return_result=True
        )
        
        if result:
            print(f"Evaluation Result: {result['success']}")
            if result.get('performance'):
                print(f"Performance Data: {result['performance']}")


if __name__ == "__main__":
    asyncio.run(main())
