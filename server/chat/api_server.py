# -*- coding: utf-8 -*-
"""
聊天服务HTTP API服务器
提供REST接口访问聊天服务
"""
import logging
from typing import Dict, Any, Optional
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from .service import ChatService
from .deprecated.models import ChatRequest, ErrorResponse, ModelName, ModelConfig, ModelProvider

# Pydantic模型用于API
class ChatRequestModel(BaseModel):
    conversation_id: str
    message: str
    model: str
    thought: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False

class ChatResponseModel(BaseModel):
    request_id: str
    conversation_id: str
    message: str
    thought: Optional[str] = None
    model: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    processing_time: Optional[float] = None

class ErrorResponseModel(BaseModel):
    request_id: str
    conversation_id: str
    error: str
    error_type: str

class ChatAPIServer:
    """聊天API服务器"""
    
    def __init__(self, 
                 chat_service: ChatService,
                 host: str = "0.0.0.0",
                 port: int = 8000,
                 logger: Optional[logging.Logger] = None):
        """初始化API服务器
        
        Args:
            chat_service: 聊天服务实例
            host: 服务器主机
            port: 服务器端口
            logger: 日志记录器
        """
        self.chat_service = chat_service
        self.host = host
        self.port = port
        self.logger = logger or logging.getLogger(__name__)
        
        # 创建FastAPI应用
        self.app = FastAPI(
            title="Chat Service API",
            description="基于生产者消费者模式的大模型聊天服务",
            version="1.0.0"
        )
        
        # 注册路由
        self._setup_routes()
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.post("/chat", response_model=ChatResponseModel)
        async def chat_endpoint(request: ChatRequestModel):
            """聊天接口"""
            try:
                # 验证模型名称
                try:
                    model = ModelName(request.model)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported model: {request.model}"
                    )
                
                # 创建聊天请求
                chat_request = ChatRequest(
                    conversation_id=request.conversation_id,
                    message=request.message,
                    model=model,
                    thought=request.thought,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stream=request.stream
                )
                
                # 发送请求
                response = await self.chat_service.chat(chat_request)
                
                # 处理响应
                if isinstance(response, ErrorResponse):
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": response.error,
                            "error_type": response.error_type,
                            "request_id": response.request_id
                        }
                    )
                
                return ChatResponseModel(
                    request_id=response.request_id,
                    conversation_id=response.conversation_id,
                    message=response.message,
                    thought=response.thought,
                    model=response.model,
                    usage=response.usage,
                    finish_reason=response.finish_reason,
                    processing_time=response.processing_time
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Chat endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/stream_chat")
        async def stream_chat_endpoint(request: ChatRequestModel):
            """流式聊天接口 - 支持genai和其他provider的流式输出"""
            try:
                # 验证模型名称
                try:
                    model = ModelName(request.model)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported model: {request.model}"
                    )
                
                # 创建聊天请求
                chat_request = ChatRequest(
                    conversation_id=request.conversation_id,
                    message=request.message,
                    model=model,
                    thought=request.thought,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stream=True  # 强制开启流式模式
                )
                
                # 创建流式响应生成器
                async def generate_stream():
                    try:
                        async for chunk in self.chat_service.stream_chat(chat_request):
                            # 以Server-Sent Events格式输出
                            yield f"data: {json.dumps({'content': chunk})}\n\n"
                        
                        # 发送结束标记
                        yield f"data: {json.dumps({'finish_reason': 'stop'})}\n\n"
                        yield "data: [DONE]\n\n"
                        
                    except Exception as e:
                        self.logger.error(f"Stream generation error: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/plain",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/event-stream"
                    }
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Stream chat endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/genai_stream_chat")
        async def genai_stream_chat_endpoint(request: ChatRequestModel):
            """专用于Gemini API的流式聊天接口"""
            try:
                # 验证模型是否为Gemini
                try:
                    model = ModelName(request.model)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported model: {request.model}"
                    )
                
                # 检查是否为Gemini模型
                if not ("gemini" in request.model.lower() or "genai" in request.model.lower()):
                    raise HTTPException(
                        status_code=400,
                        detail=f"This endpoint is only for Gemini models, got: {request.model}"
                    )
                
                # 创建聊天请求
                chat_request = ChatRequest(
                    conversation_id=request.conversation_id,
                    message=request.message,
                    model=model,
                    thought=request.thought,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stream=True
                )
                
                # 创建Gemini专用流式响应生成器
                async def generate_gemini_stream():
                    try:
                        self.logger.info(f"Starting Gemini stream for model: {request.model}")
                        
                        chunk_count = 0
                        async for chunk in self.chat_service.stream_chat(chat_request):
                            chunk_count += 1
                            
                            # 实时打印日志（可选）
                            if chunk_count <= 3:  # 只打印前几个chunk的日志
                                self.logger.debug(f"Gemini chunk {chunk_count}: {chunk[:50]}...")
                            
                            # 格式化为JSON输出
                            response_data = {
                                "content": chunk,
                                "model": request.model,
                                "chunk_id": chunk_count,
                                "finish_reason": None
                            }
                            yield f"data: {json.dumps(response_data)}\n\n"
                        
                        # 发送结束标记
                        final_data = {
                            "content": "",
                            "model": request.model,
                            "chunk_id": chunk_count + 1,
                            "finish_reason": "stop"
                        }
                        yield f"data: {json.dumps(final_data)}\n\n"
                        yield "data: [DONE]\n\n"
                        
                        self.logger.info(f"Gemini stream completed with {chunk_count} chunks")
                        
                    except Exception as e:
                        self.logger.error(f"Gemini stream generation error: {e}")
                        error_data = {
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "model": request.model
                        }
                        yield f"data: {json.dumps(error_data)}\n\n"
                
                return StreamingResponse(
                    generate_gemini_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "Content-Type"
                    }
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Gemini stream chat endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/conversations/{conversation_id}/history")
        async def get_conversation_history(conversation_id: str, model: str = "openai/gpt-4o-mini"):
            """获取对话历史"""
            try:
                history = self.chat_service.get_conversation_history(conversation_id, model)
                return {"conversation_id": conversation_id, "history": history}
            except Exception as e:
                self.logger.error(f"Get history error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats")
        async def get_stats():
            """获取服务统计信息"""
            try:
                stats = self.chat_service.get_stats()
                return stats
            except Exception as e:
                self.logger.error(f"Get stats error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            try:
                health = await self.chat_service.health_check()
                return health
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models")
        async def get_supported_models():
            """获取支持的模型列表"""
            return {
                "models": [model.value for model in ModelName],
                "providers": [provider.value for provider in ModelProvider]
            }
        
        @self.app.post("/service/start")
        async def start_service():
            """启动服务"""
            try:
                await self.chat_service.start()
                return {"message": "Service started successfully"}
            except Exception as e:
                self.logger.error(f"Start service error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/service/stop")
        async def stop_service():
            """停止服务"""
            try:
                await self.chat_service.stop()
                return {"message": "Service stopped successfully"}
            except Exception as e:
                self.logger.error(f"Stop service error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def start(self):
        """启动API服务器"""
        self.logger.info(f"Starting API server on {self.host}:{self.port}")
        
        # 启动聊天服务
        await self.chat_service.start()
        
        # 启动HTTP服务器
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop(self):
        """停止API服务器"""
        self.logger.info("Stopping API server...")
        await self.chat_service.stop()

def create_chat_api_server(model_configs: Dict[str, ModelConfig], 
                          host: str = "0.0.0.0", 
                          port: int = 8000,
                          max_workers: int = 10,
                          logger: Optional[logging.Logger] = None) -> ChatAPIServer:
    """创建聊天API服务器实例
    
    Args:
        model_configs: 模型配置字典
        host: 服务器主机
        port: 服务器端口
        max_workers: 最大工作线程数
        logger: 日志记录器
        
    Returns:
        ChatAPIServer: API服务器实例
    """
    # 创建聊天服务
    chat_service = ChatService(
        model_configs=model_configs,
        max_workers=max_workers,
        logger=logger
    )
    
    # 创建API服务器
    api_server = ChatAPIServer(
        chat_service=chat_service,
        host=host,
        port=port,
        logger=logger
    )
    
    return api_server 