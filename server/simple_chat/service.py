# -*- coding: utf-8 -*-
"""
聊天服务核心类
基于精简后的 /llm 模块的简化实现
"""
import asyncio
import logging
import time
from typing import Dict, Optional, AsyncGenerator
from collections import defaultdict

# 导入精简后的 LLM 模块
from llm.providers import (
    get_provider, PlatformType, 
    ChatRequest as LLMChatRequest, Message as LLMMessage,
    OpenRouterMessage, OpenRouterChatHistory, OpenRouterFileManager,
    GeminiMessage, GeminiChatHistory, GeminiFileManager
)

# 导入对话树
from llm.history import ConversationTree

# 导入本地模型
from .models import (
    ChatRequest, ChatResponse, ErrorResponse, ConversationHistory,
    ConversationMessage, ServiceStats, HealthStatus, SupportedModel
)

class ConversationManager:
    """对话管理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.conversations: Dict[str, Dict[str, any]] = {}  # {conversation_id: {tree, platform_type}}
    
    def _get_platform_type(self, model: str) -> PlatformType:
        """根据模型名称确定平台类型"""
        if model.startswith(("openai/", "anthropic/", "deepseek/", "google/")):
            return PlatformType.OPENROUTER
        elif model.startswith("gemini-"):
            return PlatformType.GOOGLE_OFFICIAL
        elif model in ["gpt-4o", "gpt-4o-mini"]:
            return PlatformType.OPENAI_OFFICIAL
        else:
            return PlatformType.OPENROUTER  # 默认使用 OpenRouter
    
    def _create_chat_tree(self, platform_type: PlatformType):
        """创建对话树"""
        if platform_type == PlatformType.GOOGLE_OFFICIAL:
            return ConversationTree(
                message_cls=GeminiMessage,
                history_cls=GeminiChatHistory,
                system_prompt="You are a helpful AI assistant.",
                file_manager=GeminiFileManager() if hasattr(GeminiFileManager, '__call__') else None,
                logger=self.logger
            )
        else:
            return ConversationTree(
                message_cls=OpenRouterMessage,
                history_cls=OpenRouterChatHistory,
                system_prompt="You are a helpful AI assistant.",
                file_manager=OpenRouterFileManager() if hasattr(OpenRouterFileManager, '__call__') else None,
                logger=self.logger
            )
    
    def get_or_create_conversation(self, conversation_id: str, model: str):
        """获取或创建对话"""
        if conversation_id not in self.conversations:
            platform_type = self._get_platform_type(model)
            tree = self._create_chat_tree(platform_type)
            
            self.conversations[conversation_id] = {
                'tree': tree,
                'platform_type': platform_type,
                'created_at': time.time()
            }
            self.logger.info(f"Created conversation {conversation_id} for platform {platform_type}")
        
        return self.conversations[conversation_id]
    
    def add_message(self, conversation_id: str, role: str, content: str, model: str, thoughts: Optional[str] = None):
        """添加消息到对话"""
        conv = self.get_or_create_conversation(conversation_id, model)
        tree = conv['tree']
        
        if role == "user":
            tree.add_user_message(text=content)
        elif role == "assistant":
            # 使用正确的参数名
            tree.add_assistant_message(text=content, thought=thoughts)
    
    def get_history(self, conversation_id: str) -> ConversationHistory:
        """获取对话历史"""
        if conversation_id not in self.conversations:
            return ConversationHistory(conversation_id=conversation_id)
        
        conv = self.conversations[conversation_id]
        tree = conv['tree']
        history = tree.get_history()
        
        messages = []
        for msg in history.messages:
            # 解析消息部分
            content = ""
            thoughts = None
            
            # 遍历消息的所有部分
            for part in msg.parts:
                if hasattr(part, 'content'):
                    if type(part).__name__ == 'ThoughtPart':
                        thoughts = part.content
                    elif type(part).__name__ == 'ContentPart':
                        content = part.content
                    elif type(part).__name__ == 'TextPart':
                        content = part.content
            
            if content:  # 只添加有内容的消息
                messages.append(ConversationMessage(
                    role=msg.role,
                    content=content,
                    thoughts=thoughts
                ))
        
        return ConversationHistory(
            conversation_id=conversation_id,
            messages=messages,
            created_at=conv['created_at'],
            updated_at=time.time()
        )

class ChatService:
    """聊天服务主类"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.conversation_manager = ConversationManager(logger)
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'start_time': time.time()
        }
        
        # 提供商缓存
        self._providers: Dict[PlatformType, any] = {}
    
    async def _get_provider(self, model: str):
        """获取提供商实例"""
        platform_type = self.conversation_manager._get_platform_type(model)
        
        if platform_type not in self._providers:
            try:
                provider = get_provider(platform_type)
                await provider.initialize()
                self._providers[platform_type] = provider
                self.logger.info(f"Initialized provider for {platform_type}")
            except Exception as e:
                self.logger.error(f"Failed to initialize provider {platform_type}: {e}")
                raise
        
        return self._providers[platform_type]
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """处理聊天请求"""
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # 获取对话
            conv = self.conversation_manager.get_or_create_conversation(
                request.conversation_id, request.model
            )
            
            # 添加用户消息
            self.conversation_manager.add_message(
                request.conversation_id, "user", request.message, request.model
            )
            
            # 获取提供商
            provider = await self._get_provider(request.model)
            
            # 构建LLM请求
            tree = conv['tree']
            history = tree.get_history()
            
            llm_request = LLMChatRequest(
                messages=history.to_native(),
                model=request.model,
                system_prompt=request.system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                include_thinking=request.include_thinking,
                thinking_budget=request.thinking_budget,
                reasoning_effort=request.reasoning_effort,
                stream=False  # 非流式模式
            )
            
            # 发送请求
            response = await provider.chat(llm_request)
            
            # 添加助手回复
            self.conversation_manager.add_message(
                request.conversation_id, "assistant", 
                response.content, request.model, response.thoughts
            )
            
            # 记录成功
            processing_time = time.time() - start_time
            self.stats['successful_requests'] += 1
            self.stats['response_times'].append(processing_time)
            
            return ChatResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                message=response.content,
                model=request.model,
                thoughts=response.thoughts,
                usage=response.usage,
                finish_reason=response.finish_reason,
                processing_time=processing_time
            )
            
        except Exception as e:
            # 记录失败
            self.stats['failed_requests'] += 1
            error_msg = str(e)
            self.logger.error(f"Chat request failed: {error_msg}")
            
            return ErrorResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                error=error_msg,
                error_type=type(e).__name__
            )
    
    async def stream_chat(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """流式聊天"""
        try:
            # 获取对话
            conv = self.conversation_manager.get_or_create_conversation(
                request.conversation_id, request.model
            )
            
            # 添加用户消息
            self.conversation_manager.add_message(
                request.conversation_id, "user", request.message, request.model
            )
            
            # 获取提供商
            provider = await self._get_provider(request.model)
            
            # 构建LLM请求
            tree = conv['tree']
            history = tree.get_history()
            
            llm_request = LLMChatRequest(
                messages=history.to_native(),
                model=request.model,
                system_prompt=request.system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                include_thinking=request.include_thinking,
                thinking_budget=request.thinking_budget,
                reasoning_effort=request.reasoning_effort,
                stream=True
            )
            
            # 流式响应
            full_response = ""
            full_thoughts = ""
            
            async for chunk in provider.stream_chat(llm_request):
                content = chunk.content
                
                if hasattr(chunk, 'content_type'):
                    if chunk.content_type == "thought":
                        full_thoughts += content
                    elif chunk.content_type in ["answer", "answer_start"]:
                        full_response += content
                        yield content
                    else:
                        full_response += content
                        yield content
                else:
                    # 如果没有content_type，直接当作内容处理
                    full_response += content
                    yield content
            
            # 添加完整的助手回复
            self.conversation_manager.add_message(
                request.conversation_id, "assistant", 
                full_response, request.model, full_thoughts
            )
            
        except Exception as e:
            error_msg = f"Stream error: {str(e)}"
            self.logger.error(error_msg)
            yield f"ERROR: {error_msg}"
    
    def get_conversation_history(self, conversation_id: str) -> ConversationHistory:
        """获取对话历史"""
        return self.conversation_manager.get_history(conversation_id)
    
    def get_stats(self) -> ServiceStats:
        """获取服务统计"""
        response_times = self.stats['response_times']
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        return ServiceStats(
            total_requests=self.stats['total_requests'],
            successful_requests=self.stats['successful_requests'],
            failed_requests=self.stats['failed_requests'],
            average_response_time=avg_response_time,
            active_conversations=len(self.conversation_manager.conversations),
            uptime=time.time() - self.stats['start_time']
        )
    
    async def health_check(self) -> HealthStatus:
        """健康检查"""
        providers = {}
        
        for platform_type in [PlatformType.OPENROUTER, PlatformType.GOOGLE_OFFICIAL]:
            try:
                provider = get_provider(platform_type)
                await provider.initialize()
                providers[platform_type.value] = True
            except Exception as e:
                self.logger.warning(f"Provider {platform_type} health check failed: {e}")
                providers[platform_type.value] = False
        
        return HealthStatus(
            status="healthy" if any(providers.values()) else "unhealthy",
            providers=providers
        )
    
    async def close(self):
        """关闭服务"""
        for provider in self._providers.values():
            try:
                await provider.close()
            except Exception as e:
                self.logger.warning(f"Error closing provider: {e}")
        
        self._providers.clear()
        self.logger.info("Chat service closed") 