# -*- coding: utf-8 -*-
"""
具体的LLM提供商实现模块
"""
import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional

from .base import (
    ChatRequest, ChatResponse, StreamChunk, ChatMessage,
    OpenAICompatibleProvider, ProviderError, AuthenticationError, ValidationError,
    Provider,
)
from .types import PlatformType
# 导入工厂类
from .factory import (
    ProviderFactory,
)

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
    """Google genai官方API提供商 - 支持思考功能"""
    
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
    
    def _get_generate_config(self, request: ChatRequest) -> Any:
        """获取生成配置，包含思考功能"""
        config = self._types.GenerateContentConfig()
        
        # 设置基本参数
        if request.max_tokens:
            config.max_output_tokens = request.max_tokens
        if request.temperature is not None:
            config.temperature = request.temperature
        
        # 设置思考功能
        if request.include_thinking or isinstance(request.thinking_budget, int):
            config.thinking_config = self._types.ThinkingConfig(
                thinking_budget=request.thinking_budget or -1,
                include_thoughts=True
            )
        
        return config
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """发送聊天请求"""
        async with self.ensure_initialized():
            errors = self.validate_request(request)
            if errors:
                raise ValidationError(f"Request validation failed: {', '.join(errors)}", self.platform_type)
            
            try:
                if request.include_thinking:
                    # 使用流式方式处理思考功能
                    return await self._chat_with_thinking(request)
                else:
                    # 使用标准方式
                    contents = request.messages.to_native_format()
                    config = self._get_generate_config(request)
                    
                    response = await self._client.aio.models.generate_content(
                        model=request.model,
                        contents=contents,
                        config=config
                    )
                    
                    return ChatResponse(
                        content="<answer>" + (response.text or "") + "</answer>\n",
                        model=request.model,
                        finish_reason="stop"
                    )
                
            except Exception as e:
                self.logger.error(f"Google genai request failed: {e}")
                raise self.convert_exception(e)
    
    async def _chat_with_thinking(self, request: ChatRequest) -> ChatResponse:
        """使用思考功能的聊天请求"""
        contents = request.messages.to_native_format()
        config = self._get_generate_config(request)
        
        thoughts = ""
        answer = ""
        
        for chunk in self._client.models.generate_content_stream(
            model=request.model,
            contents=contents,
            config=config,
        ):
            for part in chunk.candidates[0].content.parts:
                if not part.text:
                    continue
                elif part.thought:
                    thoughts += part.text
                else:
                    answer += part.text
        
        return ChatResponse(
            content=answer,
            model=request.model,
            finish_reason="stop",
            thoughts=thoughts if thoughts else ""
        )
    
    async def stream_chat(self, request: ChatRequest) -> AsyncGenerator[StreamChunk, None]:
        """流式聊天请求"""
        async with self.ensure_initialized():
            errors = self.validate_request(request)
            if errors:
                raise ValidationError(f"Request validation failed: {', '.join(errors)}", self.platform_type)
            
            try:
                if request.include_thinking:
                    # 使用思考功能的流式处理
                    async for chunk in self._stream_with_thinking(request):
                        yield chunk
                else:
                    # 标准流式处理
                    contents = request.messages.to_native_format()
                    config = self._get_generate_config(request)
                    
                    async for chunk in self._client.aio.models.generate_content_stream(
                        model=request.model,
                        contents=contents,
                        config=config
                    ):
                        if chunk.candidates and chunk.candidates[0].content.parts: # type: ignore
                            for part in chunk.candidates[0].content.parts: # type: ignore
                                if part.text:
                                    yield StreamChunk(
                                        content=part.text,
                                        content_type="answer"
                                    )
                                    
            except Exception as e:
                self.logger.error(f"Google genai stream request failed: {e}")
                raise self.convert_exception(e)
    
    async def _stream_with_thinking(self, request: ChatRequest) -> AsyncGenerator[StreamChunk, None]:
        """支持思考功能的流式处理"""
        contents = request.messages.to_native_format()
        config = self._get_generate_config(request)
        
        thoughts_started = False
        answer_started = False
        
        for chunk in self._client.models.generate_content_stream(
            model=request.model,
            contents=contents,
            config=config,
        ):
            for part in chunk.candidates[0].content.parts:
                if not part.text:
                    continue
                elif part.thought:
                    if not thoughts_started:
                        yield StreamChunk(
                            content="<thought>"+(part.text or ""),
                            content_type="thought_start"
                        )
                        thoughts_started = True
                    
                    yield StreamChunk(
                        content=part.text or "",
                        content_type="thought"
                    )
                else:
                    if thoughts_started and not answer_started:
                        yield StreamChunk(
                            content="</thought>\n<answer>"+(part.text or ""),
                            content_type="thought_end"
                        )
                        answer_started = True
                    
                    yield StreamChunk(
                        content=part.text or "",
                        content_type="answer"
                    )
        
        # 如果有思考内容但没有输出结束标记
        if thoughts_started and not answer_started:
            yield StreamChunk(
                content="</thought>\n<answer>",
                content_type="thought_end"
            )
        
        # 标记结束
        yield StreamChunk(
            content="</answer>\n",
            finish_reason="stop",
            content_type="end"
        )

    async def stream_chat_completion(self, request: ChatRequest) -> str:
        """流式聊天请求，返回完整内容"""
        response = ""
        async for chunk in self.stream_chat(request):
            response += chunk.content
        return response

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
