# -*- coding: utf-8 -*-
"""
聊天服务API提供商适配器
统一不同API提供商的接口
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator
import openai
import time

try:
    import google.genai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    genai = None

from .models import ChatRequest, ChatResponse, ErrorResponse, ModelConfig, ModelProvider

class ChatProvider(ABC):
    """聊天提供商抽象基类"""
    
    def __init__(self, config: ModelConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    async def chat(self, request: ChatRequest, history: list) -> ChatResponse:
        """发送聊天请求"""
        pass
    
    @abstractmethod
    async def stream_chat(self, request: ChatRequest, history: list) -> AsyncGenerator[str, None]:
        """流式聊天请求"""
        pass

class OpenRouterProvider(ChatProvider):
    """OpenRouter API提供商（支持OpenAI、Claude、DeepSeek等）"""
    
    def __init__(self, config: ModelConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    
    def _prepare_messages(self, request: ChatRequest, history: list) -> list:
        """准备消息格式"""
        messages = []
        
        # 添加历史消息
        for msg in history:
            content = []
            if hasattr(msg, 'thought') and msg.thought:
                content.append({"type": "text", "text": f"<Thought>\n{msg.thought}\n</Thought>"})
            content.append({"type": "text", "text": msg.content})
            
            messages.append({
                "role": msg.role,
                "content": content if len(content) > 1 else msg.content
            })
        
        # 添加当前请求
        content = []
        if request.thought:
            content.append({"type": "text", "text": f"<Thought>\n{request.thought}\n</Thought>"})
        content.append({"type": "text", "text": request.message})
        
        messages.append({
            "role": "user",
            "content": content if len(content) > 1 else request.message
        })
        
        return messages
    
    async def chat(self, request: ChatRequest, history: list) -> ChatResponse:
        """发送聊天请求"""
        start_time = time.time()
        
        try:
            messages = self._prepare_messages(request, history)
            
            response = await self.client.chat.completions.create(
                model=request.model.value,
                messages=messages,
                max_tokens=request.max_tokens or self.config.max_tokens,
                temperature=request.temperature or self.config.temperature,
                timeout=self.config.timeout
            )
            
            content = response.choices[0].message.content
            
            # 解析thought和content
            thought = None
            if "<Thought>" in content and "</Thought>" in content:
                thought_start = content.find("<Thought>") + 10
                thought_end = content.find("</Thought>")
                thought = content[thought_start:thought_end].strip()
                content = content[thought_end + 11:].strip()
            
            processing_time = time.time() - start_time
            
            return ChatResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                message=content,
                thought=thought,
                model=request.model.value,
                usage=response.usage.dict() if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"OpenRouter API error: {e}")
            return ErrorResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                error=str(e),
                error_type=type(e).__name__
            )
    
    async def stream_chat(self, request: ChatRequest, history: list) -> AsyncGenerator[str, None]:
        """流式聊天请求"""
        try:
            messages = self._prepare_messages(request, history)
            
            stream = await self.client.chat.completions.create(
                model=request.model.value,
                messages=messages,
                max_tokens=request.max_tokens or self.config.max_tokens,
                temperature=request.temperature or self.config.temperature,
                stream=True,
                timeout=self.config.timeout
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"OpenRouter streaming error: {e}")
            yield f"Error: {str(e)}"

class ProviderFactory:
    """提供商工厂类"""
    
    @staticmethod
    def create_provider(config: ModelConfig, logger: Optional[logging.Logger] = None) -> ChatProvider:
        """创建提供商实例 - 所有模型都通过OpenRouter"""
        # 统一使用OpenRouter提供商
        return OpenRouterProvider(config, logger) 