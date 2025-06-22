# -*- coding: utf-8 -*-
"""
聊天服务HTTP API服务器
基于精简后的 /llm 模块的简化实现
"""
import logging
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

from .service import ChatService
from .models import (
    ChatRequest, ChatResponse, ErrorResponse, ConversationHistory,
    ServiceStats, HealthStatus, SupportedModel
)

class ChatAPIServer:
    """聊天API服务器"""
    
    def __init__(self, 
                 host: str = "0.0.0.0",
                 port: int = 8001,
                 logger: Optional[logging.Logger] = None):
        """初始化API服务器"""
        self.host = host
        self.port = port
        self.logger = logger or logging.getLogger(__name__)
        
        # 创建聊天服务
        self.chat_service = ChatService(logger=self.logger)
        
        # 创建FastAPI应用
        self.app = FastAPI(
            title="Chat Service API v2.0",
            description="基于精简后LLM模块的聊天服务",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # 注册路由
        self._setup_routes()
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.post("/chat", response_model=ChatResponse)
        async def chat_endpoint(request: ChatRequest):
            """聊天接口"""
            try:
                response = await self.chat_service.chat(request)
                
                # 如果是错误响应，转为HTTP异常
                if isinstance(response, ErrorResponse):
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": response.error,
                            "error_type": response.error_type,
                            "error_id": response.error_id
                        }
                    )
                
                return response
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Chat endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/stream_chat")
        async def stream_chat_endpoint(request: ChatRequest):
            """流式聊天接口"""
            try:
                # 强制开启流式模式
                request.stream = True
                
                async def generate_stream():
                    try:
                        async for chunk in self.chat_service.stream_chat(request):
                            if chunk.startswith("ERROR:"):
                                # 发送错误并结束
                                yield f"data: {json.dumps({'error': chunk[6:].strip()})}\n\n"
                                yield "data: [DONE]\n\n"
                                return
                            else:
                                # 正常内容
                                yield f"data: {json.dumps({'content': chunk})}\n\n"
                        
                        # 发送结束标记
                        yield f"data: {json.dumps({'finish_reason': 'stop'})}\n\n"
                        yield "data: [DONE]\n\n"
                        
                    except Exception as e:
                        self.logger.error(f"Stream generation error: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/event-stream",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Stream chat endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/conversations/{conversation_id}/history", response_model=ConversationHistory)
        async def get_conversation_history(conversation_id: str):
            """获取对话历史"""
            try:
                history = self.chat_service.get_conversation_history(conversation_id)
                return history
            except Exception as e:
                self.logger.error(f"Get history error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/stats", response_model=ServiceStats)
        async def get_stats():
            """获取服务统计"""
            try:
                stats = self.chat_service.get_stats()
                return stats
            except Exception as e:
                self.logger.error(f"Get stats error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health", response_model=HealthStatus)
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
            try:
                models = [model.value for model in SupportedModel]
                return {"models": models}
            except Exception as e:
                self.logger.error(f"Get models error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # OpenAI兼容路径 (避免404)
        @self.app.get("/v1/models")
        async def get_supported_models_v1():
            """OpenAI兼容的模型列表接口"""
            try:
                models = [model.value for model in SupportedModel]
                # 转换为OpenAI格式
                openai_models = [
                    {
                        "id": model,
                        "object": "model",
                        "created": 1640995200,  # 固定时间戳
                        "owned_by": "chat-service-v2"
                    }
                    for model in models
                ]
                return {"data": openai_models, "object": "list"}
            except Exception as e:
                self.logger.error(f"Get v1 models error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # 添加CORS支持
        @self.app.middleware("http")
        async def add_cors_header(request, call_next):
            response = await call_next(request)
            response.headers["Access-Control-Allow-Origin"] = "*" # type: ignore
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS" # type: ignore
            response.headers["Access-Control-Allow-Headers"] = "*" # type: ignore
            return response
    
    async def start(self):
        """启动服务器"""
        self.logger.info(f"Starting chat API server on {self.host}:{self.port}")
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop(self):
        """停止服务器"""
        self.logger.info("Stopping chat service")
        await self.chat_service.close()

def create_chat_api_server(host: str = "0.0.0.0", 
                          port: int = 8000,
                          logger: Optional[logging.Logger] = None) -> ChatAPIServer:
    """创建聊天API服务器的便捷函数"""
    return ChatAPIServer(host=host, port=port, logger=logger) 