# -*- coding: utf-8 -*-
"""
内核开发服务API服务器
提供RESTful API接口
"""
import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid

from .service import KernelDevelopmentService
from .models import (
    CompileRequest, TestRequest, PerfRequest, LLMRequest,
    RAGRequest, WebSearchRequest, SMTRequest,
    KernelCode, TestInput, KernelType, GPUResourceConfig
)

class KernelAPIServer:
    """内核开发API服务器"""
    
    def __init__(self, 
                 host: str = "0.0.0.0",
                 port: int = 8001,
                 gpu_config: Optional[GPUResourceConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """初始化API服务器"""
        self.host = host
        self.port = port
        self.logger = logger or logging.getLogger(__name__)
        
        # 创建FastAPI应用
        self.app = FastAPI(
            title="内核开发服务API",
            description="CUDA/Triton内核编译、测试、性能测试和LLM交互服务",
            version="1.0.0"
        )
        
        # 配置CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 内核开发服务
        self.kernel_service = KernelDevelopmentService(
            gpu_config=gpu_config,
            logger=self.logger
        )
        
        # 设置路由
        self._setup_routes()
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """启动事件"""
            await self.kernel_service.start()
            self.logger.info(f"Kernel API Server starting on {self.host}:{self.port}")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """关闭事件"""
            await self.kernel_service.stop()
            self.logger.info("Kernel API Server stopped")
        
        # 健康检查
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            try:
                health = await self.kernel_service.health_check()
                return JSONResponse(content=health)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # 服务统计
        @self.app.get("/stats")
        async def get_stats():
            """获取服务统计信息"""
            try:
                stats = self.kernel_service.get_service_stats()
                return JSONResponse(content=stats)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # 对话统计
        @self.app.get("/conversations/{conversation_id}/stats")
        async def get_conversation_stats(conversation_id: str):
            """获取特定对话的统计信息"""
            try:
                stats = self.kernel_service.get_conversation_stats(conversation_id)
                return JSONResponse(content=stats)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # 编译内核
        @self.app.post("/compile")
        async def compile_kernel(request_data: Dict[str, Any]):
            """编译内核"""
            try:
                # 解析请求
                kernel_code = KernelCode(
                    kernel_type=KernelType(request_data["kernel_type"]),
                    code=request_data["code"],
                    compile_flags=request_data.get("compile_flags", [])
                )
                
                compile_request = CompileRequest(
                    conversation_id=request_data.get("conversation_id", str(uuid.uuid4())),
                    kernel_code=kernel_code,
                    build_dir=request_data.get("build_dir"),
                    verbose=request_data.get("verbose", False)
                )
                
                # 执行编译
                response = await self.kernel_service.compile_kernel(
                    compile_request, 
                    priority=request_data.get("priority", 5)
                )
                
                return self._serialize_response(response)
                
            except Exception as e:
                self.logger.error(f"Compile API error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        # 功能测试
        @self.app.post("/test")
        async def test_kernel(request_data: Dict[str, Any]):
            """功能测试内核"""
            try:
                # 解析请求
                kernel_code = KernelCode(
                    kernel_type=KernelType(request_data["kernel_type"]),
                    code=request_data["code"]
                )
                
                test_inputs = TestInput(
                    input_shapes=request_data["test_inputs"]["shapes"],
                    input_dtypes=request_data["test_inputs"]["dtypes"],
                    custom_inputs=request_data["test_inputs"].get("custom_inputs")
                )
                
                test_request = TestRequest(
                    conversation_id=request_data.get("conversation_id", str(uuid.uuid4())),
                    kernel_code=kernel_code,
                    test_inputs=test_inputs,
                    tolerance=request_data.get("tolerance", {"atol": 1e-1, "rtol": 1e-1}),
                    gpu_requirements=request_data.get("gpu_requirements", [])
                )
                
                # 执行测试
                response = await self.kernel_service.test_kernel(
                    test_request,
                    priority=request_data.get("priority", 5)
                )
                
                return self._serialize_response(response)
                
            except Exception as e:
                self.logger.error(f"Test API error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        # 性能测试
        @self.app.post("/benchmark")
        async def benchmark_kernel(request_data: Dict[str, Any]):
            """性能测试内核"""
            try:
                # 解析请求
                kernel_code = KernelCode(
                    kernel_type=KernelType(request_data["kernel_type"]),
                    code=request_data["code"]
                )
                
                reference_codes = []
                for ref_data in request_data.get("reference_codes", []):
                    ref_code = KernelCode(
                        kernel_type=KernelType(ref_data["kernel_type"]),
                        code=ref_data["code"]
                    )
                    reference_codes.append(ref_code)
                
                test_inputs = TestInput(
                    input_shapes=request_data["test_inputs"]["shapes"],
                    input_dtypes=request_data["test_inputs"]["dtypes"]
                )
                
                perf_request = PerfRequest(
                    conversation_id=request_data.get("conversation_id", str(uuid.uuid4())),
                    kernel_code=kernel_code,
                    reference_codes=reference_codes,
                    test_inputs=test_inputs,
                    benchmark_config=request_data.get("benchmark_config", {}),
                    wait_for_idle=request_data.get("wait_for_idle", True)
                )
                
                # 执行性能测试
                response = await self.kernel_service.benchmark_kernel(
                    perf_request,
                    priority=request_data.get("priority", 3)
                )
                
                return self._serialize_response(response)
                
            except Exception as e:
                self.logger.error(f"Benchmark API error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        # LLM代码生成
        @self.app.post("/generate")
        async def generate_code(request_data: Dict[str, Any]):
            """使用LLM生成代码"""
            try:
                llm_request = LLMRequest(
                    conversation_id=request_data.get("conversation_id", str(uuid.uuid4())),
                    task_type=request_data["task_type"],
                    user_message=request_data["user_message"],
                    cuda_code=request_data.get("cuda_code"),
                    triton_code=request_data.get("triton_code"),
                    error_info=request_data.get("error_info"),
                    performance_data=request_data.get("performance_data"),
                    model_name=request_data.get("model_name", "deepseek/deepseek-r1:free"),
                    max_tokens=request_data.get("max_tokens", 4096),
                    temperature=request_data.get("temperature", 0.7)
                )
                
                response = await self.kernel_service.generate_code(
                    llm_request,
                    priority=request_data.get("priority", 4)
                )
                
                return self._serialize_response(response)
                
            except Exception as e:
                self.logger.error(f"Generate API error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        # RAG文档搜索
        @self.app.post("/search/docs")
        async def search_docs(request_data: Dict[str, Any]):
            """搜索本地文档"""
            try:
                rag_request = RAGRequest(
                    conversation_id=request_data.get("conversation_id", str(uuid.uuid4())),
                    query=request_data["query"],
                    knowledge_base=request_data.get("knowledge_base", "triton_docs"),
                    top_k=request_data.get("top_k", 5),
                    similarity_threshold=request_data.get("similarity_threshold", 0.7)
                )
                
                response = await self.kernel_service.search_docs(
                    rag_request,
                    priority=request_data.get("priority", 6)
                )
                
                return self._serialize_response(response)
                
            except Exception as e:
                self.logger.error(f"Search docs API error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        # 网络搜索
        @self.app.post("/search/web")
        async def search_web(request_data: Dict[str, Any]):
            """搜索网络资源"""
            try:
                web_request = WebSearchRequest(
                    conversation_id=request_data.get("conversation_id", str(uuid.uuid4())),
                    query=request_data["query"],
                    search_type=request_data.get("search_type", "triton_docs"),
                    max_results=request_data.get("max_results", 10)
                )
                
                response = await self.kernel_service.search_web(
                    web_request,
                    priority=request_data.get("priority", 7)
                )
                
                return self._serialize_response(response)
                
            except Exception as e:
                self.logger.error(f"Search web API error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        # SMT形式化验证
        @self.app.post("/verify")
        async def verify_kernel(request_data: Dict[str, Any]):
            """形式化验证内核"""
            try:
                smt_request = SMTRequest(
                    conversation_id=request_data.get("conversation_id", str(uuid.uuid4())),
                    triton_code=request_data["triton_code"],
                    cuda_code=request_data["cuda_code"],
                    verification_properties=request_data.get("verification_properties", []),
                    timeout=request_data.get("timeout", 300)
                )
                
                response = await self.kernel_service.verify_kernel(
                    smt_request,
                    priority=request_data.get("priority", 2)
                )
                
                return self._serialize_response(response)
                
            except Exception as e:
                self.logger.error(f"Verify API error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        # CUDA到Triton的完整工作流
        @self.app.post("/workflow/cuda-to-triton")
        async def cuda_to_triton_workflow(
            request_data: Dict[str, Any], 
            background_tasks: BackgroundTasks
        ):
            """CUDA转Triton的完整工作流"""
            try:
                conversation_id = request_data.get("conversation_id", str(uuid.uuid4()))
                cuda_code = request_data["cuda_code"]
                test_inputs = request_data["test_inputs"]
                max_iterations = request_data.get("max_iterations", 3)
                
                # 启动后台任务
                task_id = str(uuid.uuid4())
                background_tasks.add_task(
                    self._run_workflow,
                    task_id,
                    conversation_id,
                    cuda_code,
                    test_inputs,
                    max_iterations
                )
                
                return JSONResponse(content={
                    "task_id": task_id,
                    "conversation_id": conversation_id,
                    "status": "started",
                    "message": "工作流已启动，请查询任务状态获取进度"
                })
                
            except Exception as e:
                self.logger.error(f"Workflow API error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        # 取消任务
        @self.app.delete("/tasks/{task_id}")
        async def cancel_task(task_id: str):
            """取消任务"""
            try:
                success = await self.kernel_service.cancel_task(task_id)
                return JSONResponse(content={"cancelled": success})
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _run_workflow(self, 
                           task_id: str,
                           conversation_id: str,
                           cuda_code: str,
                           test_inputs: dict,
                           max_iterations: int):
        """运行工作流（后台任务）"""
        try:
            responses = await self.kernel_service.cuda_to_triton_workflow(
                conversation_id, cuda_code, test_inputs, max_iterations
            )
            
            # 在实际应用中，这里应该将结果存储到某个地方供查询
            self.logger.info(f"Workflow {task_id} completed with {len(responses)} responses")
            
        except Exception as e:
            self.logger.error(f"Workflow {task_id} failed: {e}")
    
    def _serialize_response(self, response) -> Dict[str, Any]:
        """序列化响应对象"""
        if hasattr(response, '__dict__'):
            return self._serialize_dataclass(response)
        else:
            return response
    
    def _serialize_dataclass(self, obj) -> Dict[str, Any]:
        """序列化数据类对象"""
        result = {}
        for key, value in obj.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = self._serialize_dataclass(value)
            elif isinstance(value, list):
                result[key] = [
                    self._serialize_dataclass(item) if hasattr(item, '__dict__') else item
                    for item in value
                ]
            elif isinstance(value, dict):
                result[key] = {
                    k: self._serialize_dataclass(v) if hasattr(v, '__dict__') else v
                    for k, v in value.items()
                }
            elif hasattr(value, 'value'):  # 枚举类型
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    async def start(self):
        """启动API服务器"""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

# 便捷启动函数
async def run_kernel_api_server(
    host: str = "0.0.0.0",
    port: int = 8001,
    gpu_config: Optional[GPUResourceConfig] = None
):
    """运行内核API服务器"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    server = KernelAPIServer(host=host, port=port, gpu_config=gpu_config)
    await server.start()

if __name__ == "__main__":
    asyncio.run(run_kernel_api_server()) 