# -*- coding: utf-8 -*-
"""
内核开发服务主类
整合GPU管理、队列管理和各种任务处理器
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
import time
from datetime import datetime

from .models import (
    KernelRequest, KernelResponse, ErrorResponse,
    TaskType, TaskStatus, CompileRequest, TestRequest, PerfRequest,
    LLMRequest, RAGRequest, WebSearchRequest, SMTRequest,
    GPUResourceConfig
)
from .gpu_manager import GPUResourceManager
from .queue_manager import KernelQueueManager
from .processors import (
    CompilerProcessor, FunctionalTestProcessor, PerformanceTestProcessor,
    LLMProcessor, RAGProcessor, WebSearchProcessor, SMTVerifierProcessor
)

class KernelDevelopmentService:
    """内核开发服务主类"""
    
    def __init__(self,
                 gpu_config: Optional[GPUResourceConfig] = None,
                 max_workers: int = 8,
                 max_queue_size: int = 500,
                 logger: Optional[logging.Logger] = None):
        """初始化内核开发服务"""
        self.logger = logger or logging.getLogger(__name__)
        
        # GPU资源管理器
        self.gpu_manager = GPUResourceManager(
            config=gpu_config,
            logger=self.logger
        )
        
        # 队列管理器
        self.queue_manager = KernelQueueManager(
            gpu_manager=self.gpu_manager,
            max_workers=max_workers,
            max_queue_size=max_queue_size,
            logger=self.logger
        )
        
        # 任务处理器
        self.processors = {
            TaskType.COMPILE: CompilerProcessor(logger=self.logger),
            TaskType.FUNCTIONAL_TEST: FunctionalTestProcessor(logger=self.logger),
            TaskType.PERFORMANCE_TEST: PerformanceTestProcessor(logger=self.logger),
            TaskType.LLM_GENERATION: LLMProcessor(logger=self.logger),
            TaskType.RAG_QUERY: RAGProcessor(logger=self.logger),
            TaskType.WEB_SEARCH: WebSearchProcessor(logger=self.logger),
            TaskType.SMT_VERIFICATION: SMTVerifierProcessor(logger=self.logger)
        }
        
        # 服务状态
        self.running = False
        self.start_time = None
        
        # 统计信息
        self.service_stats = {
            'requests_processed': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'uptime_seconds': 0
        }
    
    async def start(self):
        """启动内核开发服务"""
        if self.running:
            return
        
        self.logger.info("Starting Kernel Development Service...")
        
        try:
            # 启动GPU资源管理器
            await self.gpu_manager.start()
            
            # 注册处理器到队列管理器
            for task_type, processor in self.processors.items():
                self.queue_manager.register_processor(task_type, processor.process)
            
            # 启动队列管理器
            await self.queue_manager.start()
            
            self.running = True
            self.start_time = time.time()
            self.logger.info("Kernel Development Service started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """停止内核开发服务"""
        if not self.running:
            return
        
        self.logger.info("Stopping Kernel Development Service...")
        
        try:
            # 停止队列管理器
            await self.queue_manager.stop()
            
            # 停止GPU资源管理器
            await self.gpu_manager.stop()
            
            self.running = False
            self.logger.info("Kernel Development Service stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping service: {e}")
    
    # 统一的任务提交接口
    async def submit_task(self, 
                         request: KernelRequest, 
                         priority: int = 5) -> KernelResponse:
        """提交任务到队列"""
        if not self.running:
            return ErrorResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                task_type=self._get_task_type(request),
                error="Service is not running",
                error_type="ServiceError"
            )
        
        start_time = time.time()
        
        try:
            # 提交到队列管理器
            response = await self.queue_manager.submit_request(request, priority)
            
            # 更新统计信息
            processing_time = time.time() - start_time
            self._update_service_stats(response, processing_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error submitting task {request.request_id}: {e}")
            return ErrorResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                task_type=self._get_task_type(request),
                error=str(e),
                error_type=type(e).__name__
            )
    
    # 特定任务类型的便捷接口
    async def compile_kernel(self, request: CompileRequest, priority: int = 5) -> KernelResponse:
        """编译内核"""
        return await self.submit_task(request, priority)
    
    async def test_kernel(self, request: TestRequest, priority: int = 5) -> KernelResponse:
        """测试内核功能"""
        return await self.submit_task(request, priority)
    
    async def benchmark_kernel(self, request: PerfRequest, priority: int = 3) -> KernelResponse:
        """性能测试内核"""
        return await self.submit_task(request, priority)
    
    async def generate_code(self, request: LLMRequest, priority: int = 4) -> KernelResponse:
        """使用LLM生成代码"""
        return await self.submit_task(request, priority)
    
    async def search_docs(self, request: RAGRequest, priority: int = 6) -> KernelResponse:
        """搜索本地文档"""
        return await self.submit_task(request, priority)
    
    async def search_web(self, request: WebSearchRequest, priority: int = 7) -> KernelResponse:
        """搜索网络资源"""
        return await self.submit_task(request, priority)
    
    async def verify_kernel(self, request: SMTRequest, priority: int = 2) -> KernelResponse:
        """形式化验证内核"""
        return await self.submit_task(request, priority)
    
    # 多轮对话工作流
    async def cuda_to_triton_workflow(self, 
                                    conversation_id: str,
                                    cuda_code: str,
                                    test_inputs: dict,
                                    max_iterations: int = 3) -> List[KernelResponse]:
        """CUDA转Triton的完整工作流"""
        responses = []
        current_cuda = cuda_code
        current_triton = None
        
        for iteration in range(max_iterations):
            self.logger.info(f"Starting iteration {iteration + 1} of CUDA to Triton workflow")
            
            # 1. 使用LLM生成Triton代码
            llm_request = LLMRequest(
                conversation_id=conversation_id,
                task_type="generate",
                user_message=f"将以下CUDA代码转换为Triton内核（第{iteration + 1}次迭代）",
                cuda_code=current_cuda,
                triton_code=current_triton,
                model_name="deepseek/deepseek-r1:free"
            )
            
            llm_response = await self.generate_code(llm_request, priority=1)
            responses.append(llm_response)
            
            if not isinstance(llm_response, ErrorResponse) and llm_response.result.generated_code:
                current_triton = llm_response.result.generated_code
                
                # 2. 编译测试Triton代码
                from .models import KernelCode, KernelType
                triton_kernel = KernelCode(
                    kernel_type=KernelType.TRITON,
                    code=current_triton
                )
                
                compile_request = CompileRequest(
                    conversation_id=conversation_id,
                    kernel_code=triton_kernel
                )
                
                compile_response = await self.compile_kernel(compile_request, priority=2)
                responses.append(compile_response)
                
                if not isinstance(compile_response, ErrorResponse) and compile_response.result.success:
                    # 3. 功能测试
                    from .models import TestInput
                    test_request = TestRequest(
                        conversation_id=conversation_id,
                        kernel_code=triton_kernel,
                        test_inputs=TestInput(
                            input_shapes=test_inputs.get('shapes', [(1024, 1024)]),
                            input_dtypes=test_inputs.get('dtypes', ['float32'])
                        )
                    )
                    
                    test_response = await self.test_kernel(test_request, priority=2)
                    responses.append(test_response)
                    
                    if not isinstance(test_response, ErrorResponse) and test_response.result.success:
                        # 4. 性能测试
                        cuda_kernel = KernelCode(
                            kernel_type=KernelType.CUDA,
                            code=current_cuda
                        )
                        
                        perf_request = PerfRequest(
                            conversation_id=conversation_id,
                            kernel_code=triton_kernel,
                            reference_codes=[cuda_kernel],
                            test_inputs=test_request.test_inputs
                        )
                        
                        perf_response = await self.benchmark_kernel(perf_request, priority=1)
                        responses.append(perf_response)
                        
                        # 检查性能是否满足要求
                        if (not isinstance(perf_response, ErrorResponse) and 
                            perf_response.result.success and
                            perf_response.result.speedup_ratios.get('main_vs_reference_0', 0) >= 0.8):
                            self.logger.info(f"Workflow completed successfully in {iteration + 1} iterations")
                            break
                
                # 如果测试失败，使用错误信息优化代码
                error_info = ""
                if isinstance(compile_response, ErrorResponse):
                    error_info += f"编译错误: {compile_response.error}\n"
                elif hasattr(compile_response.result, 'errors') and compile_response.result.errors:
                    error_info += f"编译警告: {'; '.join(compile_response.result.errors)}\n"
                
                if 'test_response' in locals() and isinstance(test_response, ErrorResponse):
                    error_info += f"测试错误: {test_response.error}\n"
                
                if error_info:
                    # 使用LLM调试代码
                    debug_request = LLMRequest(
                        conversation_id=conversation_id,
                        task_type="debug",
                        user_message="请修复以下错误",
                        triton_code=current_triton,
                        error_info=error_info,
                        model_name="deepseek/deepseek-r1:free"
                    )
                    
                    debug_response = await self.generate_code(debug_request, priority=1)
                    responses.append(debug_response)
                    
                    if (not isinstance(debug_response, ErrorResponse) and 
                        debug_response.result.generated_code):
                        current_triton = debug_response.result.generated_code
        
        return responses
    
    def _get_task_type(self, request: KernelRequest) -> TaskType:
        """获取任务类型"""
        return self.queue_manager._get_task_type(request)
    
    def _update_service_stats(self, response: KernelResponse, processing_time: float):
        """更新服务统计信息"""
        self.service_stats['requests_processed'] += 1
        
        if isinstance(response, ErrorResponse):
            self.service_stats['failed_requests'] += 1
        else:
            self.service_stats['successful_requests'] += 1
        
        # 更新平均响应时间
        current_avg = self.service_stats['average_response_time']
        total_requests = self.service_stats['requests_processed']
        self.service_stats['average_response_time'] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
        
        # 更新运行时间
        if self.start_time:
            self.service_stats['uptime_seconds'] = time.time() - self.start_time
    
    # 状态和统计信息接口
    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        stats = self.service_stats.copy()
        stats['gpu_manager'] = self.gpu_manager.get_stats()
        stats['queue_manager'] = self.queue_manager.get_stats()
        stats['running'] = self.running
        
        if self.start_time:
            stats['start_time'] = datetime.fromtimestamp(self.start_time).isoformat()
        
        return stats
    
    def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """获取特定对话的统计信息"""
        return self.queue_manager.get_conversation_stats(conversation_id)
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health = {
            'service_running': self.running,
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        try:
            # GPU管理器健康检查
            health['components']['gpu_manager'] = await self.gpu_manager.health_check() # type: ignore
        except Exception as e:
            health['components']['gpu_manager'] = {'status': 'error', 'error': str(e)} # type: ignore
        
        try:
            # 队列管理器健康检查
            health['components']['queue_manager'] = await self.queue_manager.health_check() # type: ignore
        except Exception as e:
            health['components']['queue_manager'] = {'status': 'error', 'error': str(e)} # type: ignore
        
        # 处理器健康检查
        health['components']['processors'] = { # type: ignore
            task_type.value: 'available' for task_type in self.processors.keys()
        }
        
        # 整体健康状态
        all_healthy = all(
            comp.get('status') != 'error' 
            for comp in health['components'].values() 
            if isinstance(comp, dict)
        )
        health['status'] = 'healthy' if all_healthy else 'degraded'
        
        return health
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        return await self.queue_manager.cancel_task(task_id)
    
    # 上下文管理器支持
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop() 