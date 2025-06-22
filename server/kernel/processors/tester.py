# -*- coding: utf-8 -*-
"""
功能测试处理器
负责测试内核的功能正确性
"""
import os
import sys
import time
import importlib.util
import torch
from typing import List, Dict, Any
from pathlib import Path
import traceback

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from .base import BaseProcessor
from ..models import (
    TestRequest, KernelResponse, TestResult, 
    TaskType, TaskStatus, KernelType
)

class FunctionalTestProcessor(BaseProcessor):
    """功能测试处理器"""
    
    def __init__(self, logger=None):
        super().__init__(logger)
        
        # 设置环境变量
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ['TORCH_CUDA_ARCH_LIST'] = "Hopper"
    
    async def process(self, 
                     request: TestRequest, 
                     gpu_ids: List[int], 
                     context: Dict[str, Any]) -> KernelResponse:
        """处理功能测试请求"""
        start_time = time.time()
        self.log_task_start("Functional Test", request.request_id)
        
        try:
            # 检查CUDA是否可用
            if not torch.cuda.is_available():
                raise Exception("CUDA is not available")
            
            # 设置GPU环境
            if gpu_ids:
                torch.cuda.set_device(gpu_ids[0])
            
            # 创建临时工作目录
            work_dir = self.create_temp_dir("test_")
            
            # 执行功能测试
            result = await self._run_functional_test(request, work_dir, gpu_ids)
            
            # 创建响应
            response = KernelResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                task_type=TaskType.FUNCTIONAL_TEST,
                status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
                result=result,
                processing_time=time.time() - start_time,
                gpu_info=self._get_gpu_info(gpu_ids) if gpu_ids else None
            )
            
            self.log_task_end("Functional Test", request.request_id, result.success, response.processing_time)
            return response
            
        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            self.logger.error(f"Functional test failed for request {request.request_id}: {error_msg}")
            
            return self.create_error_response(
                request, TaskType.FUNCTIONAL_TEST, error_msg, 
                "FunctionalTestError", error_traceback
            )
    
    async def _run_functional_test(self, 
                                  request: TestRequest, 
                                  work_dir: str, 
                                  gpu_ids: List[int]) -> TestResult:
        """运行功能测试"""
        start_time = time.time()
        
        try:
            # 加载要测试的内核
            test_kernel = await self._load_kernel(request.kernel_code, work_dir)
            
            # 加载参考实现（如果提供）
            reference_kernel = None
            if request.reference_code:
                reference_kernel = await self._load_kernel(request.reference_code, work_dir, "reference")
            
            # 准备测试输入
            test_inputs = self._prepare_test_inputs(request.test_inputs, gpu_ids)
            
            # 执行测试内核
            test_output = await self._execute_kernel(test_kernel, test_inputs)
            
            # 如果有参考实现，执行并比较
            if reference_kernel:
                reference_output = await self._execute_kernel(reference_kernel, test_inputs)
                correctness_passed, max_diff = self._compare_outputs(
                    test_output, reference_output, request.tolerance
                )
            else:
                # 没有参考实现时，只检查内核是否能成功执行
                correctness_passed = test_output is not None
                max_diff = 0.0
            
            # 获取内存使用情况
            memory_usage = self._get_memory_usage(gpu_ids)
            
            return TestResult(
                success=correctness_passed,
                execution_time=time.time() - start_time,
                correctness_passed=correctness_passed,
                max_difference=max_diff,
                gpu_used=gpu_ids,
                memory_usage=memory_usage
            )
            
        except Exception as e:
            return TestResult(
                success=False,
                execution_time=time.time() - start_time,
                correctness_passed=False,
                gpu_used=gpu_ids,
                errors=[str(e)]
            )
    
    async def _load_kernel(self, kernel_code, work_dir: str, prefix: str = "kernel") -> callable:
        """加载内核函数"""
        if kernel_code.kernel_type == KernelType.TRITON:
            # 加载Triton内核
            triton_file = self.write_code_to_file(
                kernel_code.code, f"{prefix}.py", work_dir
            )
            return self._compile_triton_kernel_from_file(triton_file)
        
        elif kernel_code.kernel_type == KernelType.CUDA:
            # 加载CUDA内核
            cuda_file = self.write_code_to_file(
                kernel_code.code, f"{prefix}.cu", work_dir
            )
            extension = await self._compile_cuda_extension(cuda_file, work_dir, prefix)
            return getattr(extension, 'forward')
        
        elif kernel_code.kernel_type == KernelType.PYTORCH:
            # 加载PyTorch参考实现
            torch_file = self.write_code_to_file(
                kernel_code.code, f"{prefix}.py", work_dir
            )
            return self._load_torch_reference(torch_file)
        
        else:
            raise Exception(f"Unsupported kernel type: {kernel_code.kernel_type}")
    
    def _compile_triton_kernel_from_file(self, kernel_path: str) -> callable:
        """从文件编译Triton内核"""
        spec = importlib.util.spec_from_file_location(
            Path(kernel_path).stem,
            kernel_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, 'forward')
    
    async def _compile_cuda_extension(self, cuda_file: str, build_dir: str, name: str):
        """编译CUDA扩展"""
        from torch.utils.cpp_extension import load
        
        build_path = os.path.join(build_dir, f"build_{name}")
        os.makedirs(build_path, exist_ok=True)
        
        try:
            extension = load(
                name=name,
                sources=[cuda_file],
                extra_cuda_cflags=['-O3', '--use_fast_math', 
                                 '-gencode=arch=compute_80,code=sm_80', 
                                 '-gencode=arch=compute_90,code=sm_90'],
                verbose=True,
                build_directory=build_path
            )
            return extension
        except Exception as e:
            raise Exception(f"CUDA extension compilation failed: {e}")
    
    def _load_torch_reference(self, ref_path: str):
        """加载PyTorch参考实现"""
        spec = importlib.util.spec_from_file_location("torch_ref", ref_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, 'module_fn', getattr(module, 'forward', None))
    
    def _prepare_test_inputs(self, test_inputs, gpu_ids: List[int]) -> List:
        """准备测试输入数据"""
        if test_inputs.custom_inputs:
            # 使用自定义输入
            inputs = test_inputs.custom_inputs
        else:
            # 根据形状和数据类型生成输入
            inputs = []
            for i, (shape, dtype) in enumerate(zip(test_inputs.input_shapes, test_inputs.input_dtypes)):
                if dtype.startswith('float'):
                    tensor = torch.randn(shape, dtype=getattr(torch, dtype))
                elif dtype.startswith('int'):
                    tensor = torch.randint(0, 100, shape, dtype=getattr(torch, dtype))
                else:
                    tensor = torch.randn(shape)  # 默认使用float32
                
                inputs.append(tensor)
        
        # 将输入移动到GPU
        if gpu_ids:
            device = f"cuda:{gpu_ids[0]}"
            inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in inputs]
        
        return inputs
    
    async def _execute_kernel(self, kernel_func: callable, inputs: List) -> torch.Tensor:
        """执行内核函数"""
        try:
            # 预热
            for _ in range(3):
                _ = kernel_func(*inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            # 正式执行
            output = kernel_func(*inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            return output
            
        except Exception as e:
            self.logger.error(f"Kernel execution failed: {e}")
            return None
    
    def _compare_outputs(self, output1: torch.Tensor, output2: torch.Tensor, tolerance: Dict[str, float]) -> tuple:
        """比较两个输出的差异"""
        try:
            atol = tolerance.get('atol', 1e-1)
            rtol = tolerance.get('rtol', 1e-1)
            
            # 检查形状是否匹配
            if output1.shape != output2.shape:
                return False, float('inf')
            
            # 计算最大差异
            diff = torch.abs(output1 - output2)
            max_diff = torch.max(diff).item()
            
            # 使用torch.allclose检查
            matches = torch.allclose(output1, output2, atol=atol, rtol=rtol)
            
            return matches, max_diff
            
        except Exception as e:
            self.logger.error(f"Output comparison failed: {e}")
            return False, float('inf')
    
    def _get_memory_usage(self, gpu_ids: List[int]) -> Dict[str, int]:
        """获取GPU内存使用情况"""
        memory_usage = {}
        
        for gpu_id in gpu_ids:
            try:
                torch.cuda.set_device(gpu_id)
                memory_used = torch.cuda.memory_allocated(gpu_id) // (1024 * 1024)  # MB
                memory_usage[str(gpu_id)] = memory_used
            except Exception as e:
                self.logger.warning(f"Failed to get memory usage for GPU {gpu_id}: {e}")
                memory_usage[str(gpu_id)] = 0
        
        return memory_usage
    
    def _get_gpu_info(self, gpu_ids: List[int]) -> List[Dict[str, Any]]:
        """获取GPU信息"""
        gpu_info = []
        
        for gpu_id in gpu_ids:
            try:
                props = torch.cuda.get_device_properties(gpu_id)
                info = {
                    'gpu_id': gpu_id,
                    'name': props.name,
                    'memory_total': props.total_memory // (1024 * 1024),  # MB
                    'memory_used': torch.cuda.memory_allocated(gpu_id) // (1024 * 1024),  # MB
                    'compute_capability': f"{props.major}.{props.minor}"
                }
                gpu_info.append(info)
            except Exception as e:
                self.logger.warning(f"Failed to get info for GPU {gpu_id}: {e}")
        
        return gpu_info 