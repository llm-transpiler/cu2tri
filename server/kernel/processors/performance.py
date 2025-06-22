# -*- coding: utf-8 -*-
"""
性能测试处理器
负责测试内核的性能，包括执行时间、GPU利用率、内存带宽等指标
"""
import os
import sys
import time
import torch
from typing import List, Dict, Any
from pathlib import Path
import traceback

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "tools"))

from .base import BaseProcessor
from .tester import FunctionalTestProcessor  # 复用加载内核的功能
from ..models import (
    PerfRequest, KernelResponse, PerfResult, 
    TaskType, TaskStatus, KernelType
)

try:
    from tools.performer import KernelPerfBench
    PERF_BENCH_AVAILABLE = True
except ImportError:
    PERF_BENCH_AVAILABLE = False

class PerformanceTestProcessor(BaseProcessor):
    """性能测试处理器"""
    
    def __init__(self, logger=None):
        super().__init__(logger)
        
        # 设置环境变量
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ['TORCH_CUDA_ARCH_LIST'] = "Hopper"
        
        # 继承功能测试处理器的内核加载功能
        self.functional_tester = FunctionalTestProcessor(logger)
    
    async def process(self, 
                     request: PerfRequest, 
                     gpu_ids: List[int], 
                     context: Dict[str, Any]) -> KernelResponse:
        """处理性能测试请求"""
        start_time = time.time()
        self.log_task_start("Performance Test", request.request_id)
        
        try:
            # 检查CUDA是否可用
            if not torch.cuda.is_available():
                raise Exception("CUDA is not available")
            
            # 设置GPU环境
            if gpu_ids:
                torch.cuda.set_device(gpu_ids[0])
            
            # 创建临时工作目录
            work_dir = self.create_temp_dir("perf_")
            
            # 执行性能测试
            result = await self._run_performance_test(request, work_dir, gpu_ids)
            
            # 创建响应
            response = KernelResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                task_type=TaskType.PERFORMANCE_TEST,
                status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
                result=result,
                processing_time=time.time() - start_time,
                gpu_info=self._get_gpu_info(gpu_ids) if gpu_ids else None
            )
            
            self.log_task_end("Performance Test", request.request_id, result.success, response.processing_time)
            return response
            
        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            self.logger.error(f"Performance test failed for request {request.request_id}: {error_msg}")
            
            return self.create_error_response(
                request, TaskType.PERFORMANCE_TEST, error_msg, 
                "PerformanceTestError", error_traceback
            )
    
    async def _run_performance_test(self, 
                                   request: PerfRequest, 
                                   work_dir: str, 
                                   gpu_ids: List[int]) -> PerfResult:
        """运行性能测试"""
        start_time = time.time()
        
        try:
            # 加载主要内核
            main_kernel = await self.functional_tester._load_kernel(request.kernel_code, work_dir, "main")
            
            # 加载参考内核
            reference_kernels = {}
            for i, ref_code in enumerate(request.reference_codes):
                ref_name = f"reference_{i}"
                reference_kernels[ref_name] = await self.functional_tester._load_kernel(
                    ref_code, work_dir, ref_name
                )
            
            # 准备测试输入
            test_inputs = self.functional_tester._prepare_test_inputs(request.test_inputs, gpu_ids)
            
            # 获取基准配置
            benchmark_config = request.benchmark_config or {}
            warmup_runs = benchmark_config.get('warmup_runs', 10)
            test_runs = benchmark_config.get('test_runs', 100)
            
            # 性能测试结果
            kernel_performance = {}
            
            # 测试主要内核
            main_time = await self._benchmark_kernel(
                main_kernel, test_inputs, warmup_runs, test_runs, "main_kernel"
            )
            kernel_performance["main_kernel"] = main_time
            
            # 测试参考内核
            for name, kernel in reference_kernels.items():
                ref_time = await self._benchmark_kernel(
                    kernel, test_inputs, warmup_runs, test_runs, name
                )
                kernel_performance[name] = ref_time
            
            # 计算加速比
            speedup_ratios = {}
            for ref_name, ref_time in kernel_performance.items():
                if ref_name != "main_kernel" and ref_time > 0:
                    speedup_ratios[f"main_vs_{ref_name}"] = ref_time / main_time
            
            # 获取GPU利用率和内存带宽
            gpu_utilization = await self._measure_gpu_utilization(main_kernel, test_inputs, gpu_ids)
            memory_bandwidth = await self._measure_memory_bandwidth(main_kernel, test_inputs, gpu_ids)
            
            # 计算FLOPS（如果可能）
            flops = self._estimate_flops(request.test_inputs, main_time)
            
            return PerfResult(
                success=True,
                kernel_performance=kernel_performance,
                speedup_ratios=speedup_ratios,
                gpu_utilization=gpu_utilization,
                memory_bandwidth=memory_bandwidth,
                flops=flops,
                gpu_used=gpu_ids,
                detailed_metrics={
                    'warmup_runs': warmup_runs,
                    'test_runs': test_runs,
                    'input_shapes': request.test_inputs.input_shapes,
                    'benchmark_config': benchmark_config
                }
            )
            
        except Exception as e:
            return PerfResult(
                success=False,
                kernel_performance={},
                speedup_ratios={},
                gpu_utilization={},
                memory_bandwidth={},
                gpu_used=gpu_ids,
                detailed_metrics={'error': str(e)}
            )
    
    async def _benchmark_kernel(self, 
                               kernel_func: callable, 
                               inputs: List, 
                               warmup_runs: int,
                               test_runs: int,
                               kernel_name: str) -> float:
        """基准测试单个内核"""
        try:
            # 如果有专门的性能测试工具，使用它
            if PERF_BENCH_AVAILABLE:
                return KernelPerfBench.func_perf_test_median(kernel_func, inputs)
            
            # 否则使用简单的计时方法
            return await self._simple_benchmark(kernel_func, inputs, warmup_runs, test_runs)
            
        except Exception as e:
            self.logger.error(f"Benchmark failed for {kernel_name}: {e}")
            return float('inf')
    
    async def _simple_benchmark(self, 
                               kernel_func: callable, 
                               inputs: List,
                               warmup_runs: int,
                               test_runs: int) -> float:
        """简单的基准测试实现"""
        # 预热
        for _ in range(warmup_runs):
            _ = kernel_func(*inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # 性能测试
        times = []
        for _ in range(test_runs):
            start_time = time.perf_counter()
            _ = kernel_func(*inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        # 返回中位数时间
        times.sort()
        return times[len(times) // 2]
    
    async def _measure_gpu_utilization(self, 
                                      kernel_func: callable, 
                                      inputs: List,
                                      gpu_ids: List[int]) -> Dict[int, float]:
        """测量GPU利用率"""
        utilization = {}
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            for gpu_id in gpu_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                
                # 执行内核时测量利用率
                start_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # 执行内核
                _ = kernel_func(*inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # 平均利用率
                avg_util = (start_util.gpu + end_util.gpu) / 2.0
                utilization[gpu_id] = avg_util
                
        except Exception as e:
            self.logger.warning(f"Failed to measure GPU utilization: {e}")
            # 填充默认值
            for gpu_id in gpu_ids:
                utilization[gpu_id] = 0.0
        
        return utilization
    
    async def _measure_memory_bandwidth(self, 
                                       kernel_func: callable, 
                                       inputs: List,
                                       gpu_ids: List[int]) -> Dict[int, float]:
        """测量内存带宽"""
        bandwidth = {}
        
        try:
            for gpu_id in gpu_ids:
                torch.cuda.set_device(gpu_id)
                
                # 估算数据传输量
                total_bytes = 0
                for inp in inputs:
                    if isinstance(inp, torch.Tensor):
                        total_bytes += inp.numel() * inp.element_size()
                
                # 执行内核并计时
                start_time = time.perf_counter()
                _ = kernel_func(*inputs)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                # 计算带宽 (GB/s)
                execution_time = end_time - start_time
                if execution_time > 0:
                    bandwidth_gbps = (total_bytes / (1024**3)) / execution_time
                    bandwidth[gpu_id] = bandwidth_gbps
                else:
                    bandwidth[gpu_id] = 0.0
                    
        except Exception as e:
            self.logger.warning(f"Failed to measure memory bandwidth: {e}")
            # 填充默认值
            for gpu_id in gpu_ids:
                bandwidth[gpu_id] = 0.0
        
        return bandwidth
    
    def _estimate_flops(self, test_inputs, execution_time_ms: float) -> Optional[float]:
        """估算FLOPS"""
        try:
            # 简单估算：假设是矩阵乘法操作
            if len(test_inputs.input_shapes) >= 2:
                shape1 = test_inputs.input_shapes[0]
                shape2 = test_inputs.input_shapes[1]
                
                if len(shape1) >= 2 and len(shape2) >= 2:
                    # 矩阵乘法FLOPS: 2 * M * N * K
                    m, k = shape1[-2], shape1[-1]
                    n = shape2[-1]
                    ops = 2 * m * n * k
                    
                    # 考虑批次大小
                    batch_size = 1
                    for dim in shape1[:-2]:
                        batch_size *= dim
                    
                    total_ops = ops * batch_size
                    execution_time_s = execution_time_ms / 1000.0
                    
                    if execution_time_s > 0:
                        gflops = total_ops / (execution_time_s * 1e9)
                        return gflops
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to estimate FLOPS: {e}")
            return None
    
    def _get_gpu_info(self, gpu_ids: List[int]) -> List[Dict[str, Any]]:
        """获取GPU信息"""
        return self.functional_tester._get_gpu_info(gpu_ids) 