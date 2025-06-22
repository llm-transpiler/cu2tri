# -*- coding: utf-8 -*-
"""
内核编译处理器
负责编译CUDA和Triton内核
"""
import os
import sys
import time
import importlib.util
from typing import List, Dict, Any
from pathlib import Path
import tempfile
import traceback

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from .base import BaseProcessor
from ..models import (
    CompileRequest, KernelResponse, CompileResult, 
    TaskType, TaskStatus, KernelType
)

class CompilerProcessor(BaseProcessor):
    """内核编译处理器"""
    
    def __init__(self, logger=None):
        super().__init__(logger)
        
        # 设置环境变量
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ['TORCH_CUDA_ARCH_LIST'] = "Hopper"
    
    async def process(self, 
                     request: CompileRequest, 
                     gpu_ids: List[int], 
                     context: Dict[str, Any]) -> KernelResponse:
        """处理编译请求"""
        start_time = time.time()
        self.log_task_start("Kernel Compilation", request.request_id)
        
        try:
            # 创建临时工作目录
            work_dir = self.create_temp_dir("compile_")
            
            # 编译内核
            if request.kernel_code.kernel_type == KernelType.CUDA:
                result = await self._compile_cuda_kernel(request, work_dir, gpu_ids)
            elif request.kernel_code.kernel_type == KernelType.TRITON:
                result = await self._compile_triton_kernel(request, work_dir, gpu_ids)
            else:
                raise Exception(f"Unsupported kernel type: {request.kernel_code.kernel_type}")
            
            # 创建响应
            response = KernelResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                task_type=TaskType.COMPILE,
                status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
                result=result,
                processing_time=time.time() - start_time
            )
            
            self.log_task_end("Kernel Compilation", request.request_id, result.success, response.processing_time)
            return response
            
        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            self.logger.error(f"Compilation failed for request {request.request_id}: {error_msg}")
            
            return self.create_error_response(
                request, TaskType.COMPILE, error_msg, 
                "CompilationError", error_traceback
            )
    
    async def _compile_cuda_kernel(self, 
                                  request: CompileRequest, 
                                  work_dir: str, 
                                  gpu_ids: List[int]) -> CompileResult:
        """编译CUDA内核"""
        start_time = time.time()
        
        # 写入CUDA代码
        cuda_file = self.write_code_to_file(
            request.kernel_code.code, 
            "kernel.cu", 
            work_dir
        )
        
        # 设置构建目录
        build_dir = request.build_dir or os.path.join(work_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        
        try:
            # 使用torch.utils.cpp_extension编译
            extension = await self._compile_cuda_extension(
                cuda_file, build_dir, "cuda_kernel", request.kernel_code.compile_flags
            )
            
            if extension is None:
                return CompileResult(
                    success=False,
                    build_dir=build_dir,
                    compile_time=time.time() - start_time,
                    errors=["Failed to compile CUDA extension"]
                )
            
            return CompileResult(
                success=True,
                build_dir=build_dir,
                compile_time=time.time() - start_time,
                binary_path=build_dir  # PyTorch扩展没有单独的二进制文件
            )
            
        except Exception as e:
            return CompileResult(
                success=False,
                build_dir=build_dir,
                compile_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def _compile_cuda_extension(self, 
                                    cuda_file: str, 
                                    build_dir: str, 
                                    name: str,
                                    extra_flags: List[str] = None) -> Any:
        """使用PyTorch编译CUDA扩展"""
        try:
            from torch.utils.cpp_extension import load
            
            # 默认编译标志
            default_flags = [
                '-O3', '--use_fast_math', 
                '-gencode=arch=compute_80,code=sm_80', 
                '-gencode=arch=compute_90,code=sm_90'
            ]
            
            compile_flags = default_flags + (extra_flags or [])
            
            extension = load(
                name=name,
                sources=[cuda_file],
                extra_cuda_cflags=compile_flags,
                verbose=True,
                build_directory=build_dir
            )
            
            return extension
            
        except Exception as e:
            self.logger.error(f"CUDA extension compilation failed: {e}")
            return None
    
    async def _compile_triton_kernel(self, 
                                   request: CompileRequest, 
                                   work_dir: str, 
                                   gpu_ids: List[int]) -> CompileResult:
        """编译Triton内核"""
        start_time = time.time()
        
        # 写入Triton代码
        triton_file = self.write_code_to_file(
            request.kernel_code.code, 
            "kernel.py", 
            work_dir
        )
        
        try:
            # 验证Triton内核语法
            success, error_msg = await self._validate_triton_kernel(triton_file)
            
            if not success:
                return CompileResult(
                    success=False,
                    build_dir=work_dir,
                    compile_time=time.time() - start_time,
                    errors=[error_msg]
                )
            
            return CompileResult(
                success=True,
                build_dir=work_dir,
                compile_time=time.time() - start_time,
                binary_path=triton_file
            )
            
        except Exception as e:
            return CompileResult(
                success=False,
                build_dir=work_dir,
                compile_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def _validate_triton_kernel(self, triton_file: str) -> tuple:
        """验证Triton内核语法"""
        try:
            # 尝试导入和编译Triton内核
            spec = importlib.util.spec_from_file_location(
                Path(triton_file).stem,
                triton_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 检查是否有forward函数
            if not hasattr(module, 'forward'):
                return False, "Triton kernel must have a 'forward' function"
            
            # 尝试获取内核函数
            forward_func = getattr(module, 'forward')
            if not callable(forward_func):
                return False, "'forward' must be a callable function"
            
            return True, "Triton kernel validation successful"
            
        except Exception as e:
            return False, f"Triton kernel validation failed: {str(e)}"
    
    def _get_compile_command(self, 
                           kernel_type: KernelType, 
                           source_file: str, 
                           output_dir: str,
                           flags: List[str] = None) -> List[str]:
        """获取编译命令"""
        flags = flags or []
        
        if kernel_type == KernelType.CUDA:
            # NVCC编译命令
            cmd = [
                'nvcc',
                '-O3',
                '--use_fast_math',
                '-gencode=arch=compute_80,code=sm_80',
                '-gencode=arch=compute_90,code=sm_90',
                '-shared',
                '-Xcompiler', '-fPIC',
                source_file,
                '-o', os.path.join(output_dir, 'kernel.so')
            ]
            cmd.extend(flags)
            return cmd
        
        else:
            # 对于其他类型，暂时返回空命令
            return []
    
    async def _check_dependencies(self, dependencies: List[str]) -> Dict[str, bool]:
        """检查依赖项是否可用"""
        available = {}
        
        for dep in dependencies:
            try:
                if dep == "torch":
                    import torch
                    available[dep] = torch.cuda.is_available()
                elif dep == "triton":
                    import triton
                    available[dep] = True
                elif dep == "cuda":
                    # 检查CUDA编译器
                    returncode, _, _ = await self.run_command(['nvcc', '--version'])
                    available[dep] = returncode == 0
                else:
                    # 尝试导入其他模块
                    importlib.import_module(dep)
                    available[dep] = True
            except:
                available[dep] = False
        
        return available 