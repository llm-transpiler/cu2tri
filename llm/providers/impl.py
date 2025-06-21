# -*- coding: utf-8 -*-
"""
具体的LLM提供商实现 - 简化版本
"""
from typing import List, AsyncGenerator
import logging

from .base import (
    Provider, OpenAICompatibleProvider,
    ChatRequest, ChatResponse, StreamChunk, ChatMessage,
    ProviderError, AuthenticationError
)
from .factory import ProviderFactory
from .types import PlatformType

# ============ 聚合平台实现 ============

class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter聚合平台提供商"""
    pass

# ============ 官方API实现 ============

class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI官方API提供商"""
    pass

class AnthropicProvider(OpenAICompatibleProvider):
    """Anthropic官方API提供商 (使用OpenAI兼容接口)"""
    pass

class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek官方API提供商"""
    pass

class ZhipuProvider(OpenAICompatibleProvider):
    """智谱AI官方API提供商"""
    pass

class GoogleProvider(Provider):
    """Google genai官方API提供商"""
    
    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        self._genai = None
        self._types = None
    
    async def _initialize_client(self) -> None:
        """初始化Google genai客户端"""
        try:
            import google.genai as genai
            from google.genai import types
        except ImportError:
            raise ProviderError(
                "需要安装google-genai库: pip install google-genai",
                self.platform_type
            )
        
        # 初始化客户端
        if not self.config.api_key:
            raise AuthenticationError("Google genai requires API key", self.platform_type)
        
        self._client = genai.Client(api_key=self.config.api_key)
        self._types = types
        self.logger.debug("Google genai client initialized successfully")
    
    def _prepare_contents(self, messages: List[ChatMessage]) -> List[any]:
        """准备Google genai格式的内容"""
        contents = []
        for msg in messages:
            # Google genai使用 "model" 而不是 "assistant"
            role = "model" if msg.role == "assistant" else msg.role
            contents.append(self._types.Content(
                role=role,
                parts=[self._types.Part.from_text(msg.content)]
            ))
        return contents
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """发送聊天请求"""
        async with self.ensure_initialized():
            errors = self.validate_request(request)
            if errors:
                raise ProviderError(f"Request validation failed: {', '.join(errors)}", self.platform_type)
            
            try:
                contents = self._prepare_contents(request.messages)
                
                # 构建配置
                config = self._types.GenerateContentConfig()
                if request.max_tokens:
                    config.max_output_tokens = request.max_tokens
                if request.temperature is not None:
                    config.temperature = request.temperature
                
                response = await self._client.aio.models.generate_content(
                    model=request.model,
                    contents=contents,
                    config=config
                )
                
                return ChatResponse(
                    content=response.text or "",
                    model=request.model,
                    finish_reason="stop"
                )
                
            except Exception as e:
                self.logger.error(f"Google genai request failed: {e}")
                raise self._convert_exception(e)
    
    async def stream_chat(self, request: ChatRequest) -> AsyncGenerator[StreamChunk, None]:
        """流式聊天请求"""
        async with self.ensure_initialized():
            errors = self.validate_request(request)
            if errors:
                raise ProviderError(f"Request validation failed: {', '.join(errors)}", self.platform_type)
            
            try:
                contents = self._prepare_contents(request.messages)
                
                # 构建配置
                config = self._types.GenerateContentConfig()
                if request.max_tokens:
                    config.max_output_tokens = request.max_tokens
                if request.temperature is not None:
                    config.temperature = request.temperature
                
                async for chunk in self._client.aio.models.generate_content_stream(
                    model=request.model,
                    contents=contents,
                    config=config
                ):
                    if chunk.candidates and chunk.candidates[0].content.parts: # type: ignore
                        for part in chunk.candidates[0].content.parts: # type: ignore
                            if part.text:
                                yield StreamChunk(content=part.text)
                                
            except Exception as e:
                self.logger.error(f"Google genai stream request failed: {e}")
                raise self._convert_exception(e)
    
    def _convert_exception(self, e: Exception) -> ProviderError:
        """转换异常为统一错误类型"""
        error_str = str(e).lower()
        
        if "unauthorized" in error_str or "invalid api key" in error_str:
            return AuthenticationError(f"Authentication failed: {e}", self.platform_type)
        else:
            return ProviderError(f"Request failed: {e}", self.platform_type)

# ============ 本地部署实现 ============

class VLLMProvider(OpenAICompatibleProvider):
    """vLLM本地推理服务提供商"""
    pass

# ============ 自动注册提供商 ============

def _register_all_providers():
    """自动注册所有提供商到工厂"""
    providers_to_register = [
        # 聚合平台
        (PlatformType.OPENROUTER, OpenRouterProvider),
        
        # 官方API
        (PlatformType.OPENAI_OFFICIAL, OpenAIProvider),
        (PlatformType.ANTHROPIC_OFFICIAL, AnthropicProvider),
        (PlatformType.GOOGLE_OFFICIAL, GoogleProvider),
        (PlatformType.DEEPSEEK_OFFICIAL, DeepSeekProvider),
        (PlatformType.ZHIPU_OFFICIAL, ZhipuProvider),
        
        # 本地部署
        (PlatformType.VLLM, VLLMProvider),
    ]
    
    logger = logging.getLogger(__name__)
    registered_count = 0
    
    for platform_type, provider_class in providers_to_register:
        try:
            ProviderFactory.register_provider(platform_type, provider_class)
            registered_count += 1
        except Exception as e:
            logger.warning(f"Failed to register provider {platform_type.value}: {e}")
    
    logger.info(f"Successfully registered {registered_count} providers")

# 模块导入时自动注册
_register_all_providers()

# ============ 导出 ============

__all__ = [
    'OpenRouterProvider',
    'OpenAIProvider', 
    'AnthropicProvider',
    'DeepSeekProvider',
    'ZhipuProvider',
    'GoogleProvider',
    'VLLMProvider',
]
