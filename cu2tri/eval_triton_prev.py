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
import importlib.util
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import gc

# 设置CUDA环境变量
os.environ["TORCH_USE_CUDA_DSA"] = "1" 
os.environ['TORCH_CUDA_ARCH_LIST'] = "Hopper"

# 导入项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# 添加项目根目录到Python路径
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import torch
    from tools.performer import KernelPerfBench
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保安装了必要的依赖包")
    sys.exit(1)


class TritonKernelEvaluator:
    """Triton内核评估器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """初始化评估器
        
        Args:
            logger: 日志记录器，如果未提供则创建默认记录器
        """
        self.logger = logger or self._create_default_logger()
        self.tolerance = {"atol": 1e-1, "rtol": 1e-1}
        
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
    
    def set_tolerance(self, atol: float = 1e-1, rtol: float = 1e-1) -> None:
        """设置数值比较容差
        
        Args:
            atol: 绝对容差
            rtol: 相对容差
        """
        self.tolerance = {"atol": atol, "rtol": rtol}
        self.logger.info(f"设置容差: atol={atol}, rtol={rtol}")
    
    def compile_cuda_kernel_from_file(self, kernel_path: str, verbose: bool = False) -> callable:
        """从文件编译CUDA内核
        
        Args:
            kernel_path: 内核文件路径
            verbose: 是否显示详细信息
            
        Returns:
            编译后的内核函数
            
        Raises:
            ImportError: 当无法导入模块时
            AttributeError: 当找不到forward函数时
        """
        try:
            spec = importlib.util.spec_from_file_location(
                Path(kernel_path).stem,
                kernel_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"无法加载模块规范: {kernel_path}")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'forward'):
                raise AttributeError(f"模块中未找到forward函数: {kernel_path}")
                
            return getattr(module, 'forward')
            
        except Exception as e:
            self.logger.error(f"编译内核失败: {e}")
            raise
    
    def compile_cuda_extension(
        self, 
        cuda_file: str, 
        build_dir: str = './build', 
        name: str = "cuda_kernel"
    ) -> Any:
        """编译CUDA扩展
        
        Args:
            cuda_file: CUDA源文件路径
            build_dir: 构建目录
            name: 扩展名称
            
        Returns:
            编译后的扩展对象
            
        Raises:
            RuntimeError: 当编译失败时
        """
        try:
            from torch.utils.cpp_extension import load
            
            if not os.path.exists(build_dir):
                os.makedirs(build_dir)
                
            extension = load(
                name=name,
                sources=[cuda_file],
                extra_cuda_cflags=[
                    '-O3', 
                    '--use_fast_math', 
                    '-gencode=arch=compute_80,code=sm_80', 
                    '-gencode=arch=compute_90,code=sm_90'
                ],
                verbose=True,
                build_directory=build_dir
            )
            return extension
            
        except Exception as e:
            self.logger.error(f"CUDA扩展编译失败: {e}")
            raise RuntimeError(f"CUDA扩展编译失败: {e}")
    
    def load_torch_ref(self, ref_path: str) -> Any:
        """加载PyTorch参考实现
        
        Args:
            ref_path: 参考实现文件路径
            
        Returns:
            加载的模块对象
        """
        try:
            spec = importlib.util.spec_from_file_location("torch_ref", ref_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法加载PyTorch参考模块: {ref_path}")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
            
        except Exception as e:
            self.logger.error(f"加载PyTorch参考失败: {e}")
            raise
    
    def benchmark_kernel(self, kernel_func: callable, inputs: List[torch.Tensor]) -> float:
        """对内核进行性能基准测试
        
        Args:
            kernel_func: 内核函数
            inputs: 输入张量列表
            
        Returns:
            平均执行时间（毫秒）
        """
        try:
            return KernelPerfBench.func_perf_test_median(kernel_func, inputs)
        except Exception as e:
            self.logger.error(f"性能测试失败: {e}")
            return float('inf')
    
    def check_cuda_availability(self) -> bool:
        """检查CUDA是否可用"""
        if not torch.cuda.is_available():
            self.logger.error("❌ CUDA不可用，跳过测试")
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
            self.logger.info(f"📊 测试矩阵大小: {cuda_inputs[0].shape}")
            
        return cuda_inputs
    
    def compare_results(
        self, 
        triton_result: torch.Tensor,
        cuda_result: torch.Tensor, 
        torch_result: torch.Tensor
    ) -> Dict[str, bool]:
        """比较不同实现的结果
        
        Args:
            triton_result: Triton结果
            cuda_result: CUDA结果
            torch_result: PyTorch结果
            
        Returns:
            包含比较结果的字典
        """
        atol, rtol = self.tolerance["atol"], self.tolerance["rtol"]
        
        comparisons = {
            "triton_cuda_match": torch.allclose(triton_result, cuda_result, atol=atol, rtol=rtol),
            "triton_torch_match": torch.allclose(triton_result, torch_result, atol=atol, rtol=rtol),
            "cuda_torch_match": torch.allclose(cuda_result, torch_result, atol=atol, rtol=rtol)
        }
        
        self.logger.info(f"🔍 Triton vs CUDA 匹配: {'✅' if comparisons['triton_cuda_match'] else '❌'}")
        self.logger.info(f"🔍 Triton vs PyTorch 匹配: {'✅' if comparisons['triton_torch_match'] else '❌'}")
        self.logger.info(f"🔍 CUDA vs PyTorch 匹配: {'✅' if comparisons['cuda_torch_match'] else '❌'}")
        
        if not comparisons["triton_cuda_match"]:
            max_diff = torch.max(torch.abs(triton_result - cuda_result)).item()
            self.logger.info(f"   最大差异: {max_diff:.2e}")
            
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
        self.logger.info(f"测试开始时间: {current_time}{round_info}")
        self.logger.info(f"{'='*60}")
    
    def log_test_end(self) -> None:
        """记录测试结束信息"""
        self.logger.info("\n✅ 测试完成!")
        self.logger.info(f"测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'='*60}\n")
    
    def cleanup_gpu_memory(self) -> None:
        """清理GPU内存"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            self.logger.warning(f"清理GPU内存时出错: {e}")
    
    async def evaluate_triton_kernel(
        self,
        base_dir: str,
        model_name: str = "triton_test",
        time_str: Optional[str] = None,
        logfile_prefix: str = "",
        timestamp_log_dir: Optional[str] = None,
        return_result: bool = False
    ) -> Optional[Dict[str, Any]]:
        """评估Triton内核
        
        Args:
            base_dir: 基础目录路径
            model_name: 模型名称
            time_str: 时间戳字符串
            logfile_prefix: 日志文件前缀
            timestamp_log_dir: 时间戳日志目录
            return_result: 是否返回结果字典
            
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
            # 检查CUDA可用性
            if not self.check_cuda_availability():
                result = {"success": False, "error": "CUDA不可用"}
                return result if return_result else None
            
            # 加载测试数据
            self.logger.info("📁 加载测试用例...")
            torch_ref = self.load_torch_ref(torch_ref_file)
            cuda_inputs = self.prepare_test_inputs(torch_ref)
            
            # 加载Triton内核
            self.logger.info(f"⚡ 加载Triton内核: {triton_file}")
            triton_func = self.compile_cuda_kernel_from_file(triton_file, verbose=False)
            self.logger.info("✅ Triton内核加载成功")
            
            # 编译CUDA内核
            self.logger.info("🔧 编译CUDA内核...")
            cuda_extension = self.compile_cuda_extension(cuda_file, build_dir=f"{base_dir}/build")
            cuda_func = getattr(cuda_extension, 'forward')
            self.logger.info("✅ CUDA内核编译成功")
            
            # 功能正确性测试
            self.logger.info("\n🧪 开始功能正确性测试...")
            triton_result = triton_func(*cuda_inputs)
            cuda_result = cuda_func(*cuda_inputs)
            torch_result = torch_ref.module_fn(*cuda_inputs)
            
            # 比较结果
            comparisons = self.compare_results(triton_result, cuda_result, torch_result)
            
            if not (comparisons["triton_cuda_match"] and comparisons["cuda_torch_match"]):
                self.logger.info("❌ 功能测试失败，跳过性能测试")
                result = {"success": False, "error": "功能测试失败：结果不匹配"}
                return result if return_result else None
            
            # 性能测试
            self.logger.info("\n⏱️  开始性能测试...")
            triton_time = self.benchmark_kernel(triton_func, cuda_inputs)
            self.logger.info("测试CUDA内核性能...")
            cuda_time = self.benchmark_kernel(cuda_func, cuda_inputs)
            self.logger.info("测试PyTorch参考性能...")
            torch_time = self.benchmark_kernel(torch_ref.module_fn, cuda_inputs)
            
            # 性能结果
            self.logger.info(f"\n📈 性能测试结果:")
            self.logger.info(f"   Triton:   {triton_time:.3f} ms")
            self.logger.info(f"   CUDA:     {cuda_time:.3f} ms")
            self.logger.info(f"   PyTorch:  {torch_time:.3f} ms")
            
            # 计算加速比
            triton_cuda_speedup = cuda_time / triton_time if triton_time > 0 else 0.0
            triton_torch_speedup = torch_time / triton_time if triton_time > 0 else 0.0
            
            self.logger.info(f"\n🚀 加速比:")
            self.logger.info(f"   Triton vs CUDA: {triton_cuda_speedup:.2f}x")
            self.logger.info(f"   Triton vs PyTorch: {triton_torch_speedup:.2f}x")
            
            self.log_test_end()
            
            if return_result:
                return {
                    "success": True,
                    "performance": {
                        "triton_time": triton_time,
                        "cuda_time": cuda_time,
                        "torch_time": torch_time,
                        "triton_cuda_speedup": triton_cuda_speedup,
                        "triton_torch_speedup": triton_torch_speedup
                    },
                    "correctness": comparisons
                }
            
        except Exception as e:
            self.logger.error(f"❌ 评估过程出错: {e}")
            import traceback
            result = {
                "success": False, 
                "error": str(e), 
                "traceback": traceback.format_exc()
            }
            return result if return_result else None
        
        finally:
            self.cleanup_gpu_memory()


# 向后兼容的函数接口
def eval_triton_kernel(
    base_dir: str,
    model_name: str = "new_test",
    logger: Optional[logging.Logger] = None,
    time_str: Optional[str] = None,
    logfile_prefix: str = "",
    return_result: bool = False,
    timestamp_log_dir: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """评估Triton内核（同步版本，向后兼容）
    
    Args:
        base_dir: 基础目录路径
        model_name: 模型名称
        logger: 日志记录器
        time_str: 时间戳字符串
        logfile_prefix: 日志文件前缀
        return_result: 是否返回结果字典
        timestamp_log_dir: 时间戳日志目录
        
    Returns:
        如果return_result为True，返回包含评估结果的字典
    """
    evaluator = TritonKernelEvaluator(logger)
    
    # 由于原来是同步函数，这里使用asyncio.run来运行异步版本
    return asyncio.run(evaluator.evaluate_triton_kernel(
        base_dir=base_dir,
        model_name=model_name,
        time_str=time_str,
        logfile_prefix=logfile_prefix,
        timestamp_log_dir=timestamp_log_dir,
        return_result=return_result
    ))


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
            print(f"评估结果: {result['success']}")
            if result.get('performance'):
                print(f"性能数据: {result['performance']}")


if __name__ == "__main__":
    asyncio.run(main())
