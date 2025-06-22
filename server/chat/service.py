# -*- coding: utf-8 -*-
"""
聊天服务主类
整合队列管理、对话历史和API提供商
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
import time
from collections import defaultdict
import threading

from .deprecated.models import (
    ChatRequest, ChatResponse, ErrorResponse, Message,
    ModelConfig, ModelProvider, ModelName
)
from .providers import ProviderFactory, ChatProvider
from .queue_manager import ChatQueueManager

# 导入LLM库
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from llm.history.tree import ConversationTree

class ConversationManager:
    """对话管理器，管理多个对话的历史记录"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.conversations: Dict[str, ConversationTree] = {}
        self.conversation_lock = threading.RLock()
    
    def get_or_create_conversation(self, conversation_id: str, model_config: ModelConfig) -> ConversationTree:
        """获取或创建对话树"""
        with self.conversation_lock:
            if conversation_id not in self.conversations:
                # 根据模型类型选择合适的消息类和历史类
                if model_config.provider == ModelProvider.GEMINI:
                    message_cls = GeminiMessage
                    history_cls = GeminiChatHistory
                else:
                    message_cls = OpenRouterMessage
                    history_cls = OpenRouterChatHistory
                
                conversation = ConversationTree(
                    message_cls=message_cls,
                    history_cls=history_cls,
                    system_prompt="You are a helpful AI assistant.",
                    logger=self.logger
                )
                self.conversations[conversation_id] = conversation
                self.logger.info(f"Created new conversation: {conversation_id}")
            
            return self.conversations[conversation_id]
    
    def add_message_to_conversation(self, conversation_id: str, message: Message, model_config: ModelConfig):
        """添加消息到对话"""
        conversation = self.get_or_create_conversation(conversation_id, model_config)
        
        with self.conversation_lock:
            conversation.add_message(
                role=message.role,
                text=message.content,
                thought=message.thought
            )
    
    def get_conversation_history(self, conversation_id: str, model_config: ModelConfig) -> List[Message]:
        """获取对话历史"""
        conversation = self.get_or_create_conversation(conversation_id, model_config)
        
        with self.conversation_lock:
            history = conversation.get_history()
            messages = []
            
            for msg in history.messages:
                # 解析消息内容
                content = ""
                thought = None
                
                for part in msg.parts:
                    if hasattr(part, 'content') and part.content:
                        if type(part).__name__ == 'ThoughtPart':
                            thought = part.content
                        elif type(part).__name__ == 'ContentPart':
                            content = part.content
                
                if content:  # 只添加有内容的消息
                    messages.append(Message(
                        role=msg.role,
                        content=content,
                        thought=thought
                    ))
            
            return messages
    
    def get_stats(self) -> Dict[str, Any]:
        """获取对话统计信息"""
        with self.conversation_lock:
            return {
                'total_conversations': len(self.conversations),
                'conversation_ids': list(self.conversations.keys())
            }

class ChatService:
    """聊天服务主类"""
    
    def __init__(self, 
                 model_configs: Dict[str, ModelConfig],
                 max_workers: int = 10,
                 max_queue_size: int = 1000,
                 logger: Optional[logging.Logger] = None):
        """初始化聊天服务
        
        Args:
            model_configs: 模型配置字典，key为模型名称
            max_workers: 最大工作线程数
            max_queue_size: 最大队列大小
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model_configs = model_configs
        
        # 创建提供商实例
        self.providers: Dict[str, ChatProvider] = {}
        for model_name, config in model_configs.items():
            try:
                provider = ProviderFactory.create_provider(config, self.logger)
                self.providers[model_name] = provider
                self.logger.info(f"Initialized provider for model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize provider for {model_name}: {e}")
        
        # 创建对话管理器
        self.conversation_manager = ConversationManager(self.logger)
        
        # 创建队列管理器
        self.queue_manager = ChatQueueManager(
            max_workers=max_workers,
            max_queue_size=max_queue_size,
            logger=self.logger
        )
        
        # 设置请求处理器
        self.queue_manager.set_request_processor(self._process_chat_request)
        
        # 服务状态
        self.running = False
    
    async def start(self):
        """启动聊天服务"""
        if self.running:
            return
        
        self.logger.info("Starting chat service...")
        await self.queue_manager.start()
        self.running = True
        self.logger.info("Chat service started successfully")
    
    async def stop(self):
        """停止聊天服务"""
        if not self.running:
            return
        
        self.logger.info("Stopping chat service...")
        await self.queue_manager.stop()
        self.running = False
        self.logger.info("Chat service stopped")
    
    async def chat(self, request: ChatRequest, priority: int = 0) -> ChatResponse:
        """发送聊天请求
        
        Args:
            request: 聊天请求
            priority: 优先级（数字越小优先级越高）
            
        Returns:
            ChatResponse: 聊天响应或错误响应
        """
        if not self.running:
            return ErrorResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                error="Chat service is not running",
                error_type="ServiceError"
            )
        
        try:
            response = await self.queue_manager.submit_request(request, priority)
            return response
        except Exception as e:
            self.logger.error(f"Error processing chat request {request.request_id}: {e}")
            return ErrorResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                error=str(e),
                error_type=type(e).__name__
            )
    
    async def _process_chat_request(self, request: ChatRequest) -> ChatResponse:
        """处理聊天请求（内部方法）"""
        start_time = time.time()
        
        try:
            # 检查模型是否支持
            model_name = request.model.value
            if model_name not in self.providers:
                return ErrorResponse(
                    request_id=request.request_id,
                    conversation_id=request.conversation_id,
                    error=f"Model {model_name} not supported",
                    error_type="ModelNotSupported"
                )
            
            # 获取提供商和模型配置
            provider = self.providers[model_name]
            model_config = self.model_configs[model_name]
            
            # 获取对话历史
            history = self.conversation_manager.get_conversation_history(
                request.conversation_id, 
                model_config
            )
            
            # 添加用户消息到历史
            user_message = Message(
                role="user",
                content=request.message,
                thought=request.thought
            )
            self.conversation_manager.add_message_to_conversation(
                request.conversation_id,
                user_message,
                model_config
            )
            
            # 调用API
            response = await provider.chat(request, history)
            
            # 如果是成功响应，添加到对话历史
            if isinstance(response, ChatResponse):
                assistant_message = Message(
                    role="assistant",
                    content=response.message,
                    thought=response.thought
                )
                self.conversation_manager.add_message_to_conversation(
                    request.conversation_id,
                    assistant_message,
                    model_config
                )
                
                # 更新处理时间
                if response.processing_time is None:
                    response.processing_time = time.time() - start_time
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in _process_chat_request: {e}")
            return ErrorResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                error=str(e),
                error_type=type(e).__name__
            )
    
    def get_conversation_history(self, conversation_id: str, model_name: str) -> List[Message]:
        """获取对话历史"""
        if model_name not in self.model_configs:
            return []
        
        model_config = self.model_configs[model_name]
        return self.conversation_manager.get_conversation_history(conversation_id, model_config)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        queue_stats = self.queue_manager.get_stats()
        conversation_stats = self.conversation_manager.get_stats()
        
        return {
            'service_running': self.running,
            'supported_models': list(self.model_configs.keys()),
            'queue': queue_stats,
            'conversations': conversation_stats
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy' if self.running else 'stopped',
            'timestamp': time.time(),
            'stats': self.get_stats()
        }

    async def stream_chat(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """流式聊天请求 - 支持genai和其他provider
        
        Args:
            request: 聊天请求
            
        Yields:
            str: 流式响应文本块
        """
        if not self.running:
            yield "Error: Chat service is not running"
            return
        
        try:
            # 检查模型是否支持
            model_name = request.model.value
            if model_name not in self.providers:
                yield f"Error: Model {model_name} not supported"
                return
            
            # 获取提供商和模型配置
            provider = self.providers[model_name]
            model_config = self.model_configs[model_name]
            
            # 获取对话历史
            history = self.conversation_manager.get_conversation_history(
                request.conversation_id, 
                model_config
            )
            
            # 添加用户消息到历史
            user_message = Message(
                role="user",
                content=request.message,
                thought=request.thought
            )
            self.conversation_manager.add_message_to_conversation(
                request.conversation_id,
                user_message,
                model_config
            )
            
            # 收集完整响应用于历史记录
            full_response_parts = []
            full_thought = None
            
            # 流式调用API
            async for chunk in provider.stream_chat(request, history):
                # 解析可能的thought标签
                if "<Thought>" in chunk and "</Thought>" in chunk:
                    thought_start = chunk.find("<Thought>") + 10
                    thought_end = chunk.find("</Thought>")
                    thought_content = chunk[thought_start:thought_end].strip()
                    if thought_content:
                        full_thought = thought_content
                    # 移除thought标签，只yield内容部分
                    chunk = chunk[thought_end + 11:].strip()
                
                if chunk:  # 只yield非空内容
                    full_response_parts.append(chunk)
                    yield chunk
            
            # 将完整响应添加到对话历史
            full_response = "".join(full_response_parts)
            if full_response:
                assistant_message = Message(
                    role="assistant",
                    content=full_response,
                    thought=full_thought
                )
                self.conversation_manager.add_message_to_conversation(
                    request.conversation_id,
                    assistant_message,
                    model_config
                )
            
        except Exception as e:
            self.logger.error(f"Error in stream_chat: {e}")
            yield f"Error: {str(e)}" 