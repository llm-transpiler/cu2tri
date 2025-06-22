# -*- coding: utf-8 -*-
"""
基础任务处理器
定义所有处理器的公共接口和基础功能
"""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import tempfile
import shutil

from ..models import KernelRequest, KernelResponse, ErrorResponse, TaskType

class BaseProcessor(ABC):
    """基础任务处理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """初始化处理器"""
        self.logger = logger or logging.getLogger(__name__)
        self.temp_dirs = []  # 临时目录列表，用于清理
    
    @abstractmethod
    async def process(self, 
                     request: KernelRequest, 
                     gpu_ids: List[int], 
                     context: Dict[str, Any]) -> KernelResponse:
        """处理请求的抽象方法"""
        pass
    
    def create_temp_dir(self, prefix: str = "kernel_task_") -> str:
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup_temp_dirs(self):
        """清理临时目录"""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to remove temp dir {temp_dir}: {e}")
        self.temp_dirs.clear()
    
    def write_code_to_file(self, code: str, filename: str, directory: str) -> str:
        """将代码写入文件"""
        file_path = os.path.join(directory, filename)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        return file_path
    
    async def run_command(self, command: List[str], cwd: str = None, timeout: float = 300) -> tuple:
        """运行系统命令"""
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            
            return process.returncode, stdout.decode('utf-8'), stderr.decode('utf-8')
            
        except asyncio.TimeoutError:
            if 'process' in locals():
                process.kill()
                await process.wait()
            raise Exception(f"Command timed out after {timeout} seconds")
        except Exception as e:
            raise Exception(f"Failed to run command {' '.join(command)}: {e}")
    
    def get_gpu_env_vars(self, gpu_ids: List[int]) -> Dict[str, str]:
        """获取GPU环境变量"""
        env_vars = os.environ.copy()
        
        if gpu_ids:
            # 设置CUDA_VISIBLE_DEVICES
            env_vars['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
            self.logger.debug(f"Set CUDA_VISIBLE_DEVICES to {env_vars['CUDA_VISIBLE_DEVICES']}")
        
        return env_vars
    
    def log_task_start(self, task_name: str, request_id: str):
        """记录任务开始"""
        self.logger.info(f"Starting {task_name} for request {request_id}")
    
    def log_task_end(self, task_name: str, request_id: str, success: bool, duration: float):
        """记录任务结束"""
        status = "completed" if success else "failed"
        self.logger.info(f"{task_name} {status} for request {request_id} in {duration:.2f}s")
    
    def create_error_response(self, 
                            request: KernelRequest, 
                            task_type: TaskType,
                            error: str,
                            error_type: str = "ProcessingError",
                            traceback: str = None) -> ErrorResponse:
        """创建错误响应"""
        return ErrorResponse(
            request_id=request.request_id,
            conversation_id=request.conversation_id,
            task_type=task_type,
            error=error,
            error_type=error_type,
            traceback=traceback
        )
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口，自动清理资源"""
        self.cleanup_temp_dirs()
        
        if exc_type is not None:
            self.logger.error(f"Processor exited with exception: {exc_val}")
        
        return False  # 不抑制异常 