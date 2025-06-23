# -*- coding: utf-8 -*-
"""
具体的LLM提供商实现模块
"""
import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional
import time

from .base import (
    ChatRequest, ChatResponse, StreamChunk, Message, ChatHistory, FileManager,
    ProviderError, AuthenticationError, ValidationError,
    Provider,
)
from .types import PlatformType
# 导入工厂类
from .factory import (
    ProviderFactory,
)
from ..history.tree import ConversationTree
from .multimodal import GeminiMessage, OpenRouterMessage, OpenRouterChatHistory, GeminiChatHistory

class OpenRouterChatTree(ConversationTree):
    """OpenAI聊天树"""
    def __init__(self, message_cls: Message = OpenRouterMessage, history_cls: ChatHistory = OpenRouterChatHistory, 
                 system_prompt: Optional[str] = "You are a helpful assistant.", file_manager: Optional[FileManager] = None, 
                 logger: Optional[logging.Logger] = None):
        super().__init__(message_cls, history_cls, system_prompt, file_manager, logger)

class GeminiChatTree(ConversationTree):
    """Gemini聊天树"""
    def __init__(self, message_cls: Message = GeminiMessage, history_cls: ChatHistory = GeminiChatHistory, 
                 system_prompt: Optional[str] = None, file_manager: Optional[FileManager] = None, 
                 logger: Optional[logging.Logger] = None):
        super().__init__(message_cls, history_cls, system_prompt, file_manager, logger)

# ============ OpenAI兼容提供商基类 ============

class OpenAICompatibleProvider(Provider):
    """OpenAI兼容的提供商基类 - 支持思考功能
    
    大多数聚合平台和本地推理服务都兼容OpenAI API
    支持OpenAI o1系列模型的思考功能
    """
    
    async def _initialize_client(self) -> None:
        """初始化OpenAI兼容客户端"""
        try:
            import openai
        except ImportError:
            raise ProviderError("Need to install openai library: pip install openai", self.platform_type)
        
        # 获取配置
        from .config import DEFAULT_TIMEOUT
        
        client_kwargs = {
            "timeout": getattr(self.config, 'timeout', DEFAULT_TIMEOUT),
            "api_key": getattr(self.config, 'api_key', "dummy-key"),
        }
        
        # 设置base_url
        if hasattr(self.config, 'api_base') and self.config.api_base:
            client_kwargs["base_url"] = self.config.api_base
        else:
            # 从注册中心获取默认base_url
            from .registry import get_provider_registry
            registry = get_provider_registry()
            platform_info = registry.get_platform_info(self.platform_type)
            if platform_info and platform_info.base_url:
                client_kwargs["base_url"] = platform_info.base_url
        
        self._client = openai.AsyncOpenAI(**client_kwargs)
        self.logger.debug(f"OpenAI client initialized: base_url={client_kwargs.get('base_url', 'https://api.openai.com/v1')}")
    
    # def _is_reasoning_model(self, model: str) -> bool:
    #     """检查是否为支持推理的模型"""
    #     if not model:
    #         return False
        
    #     reasoning_models = ['o1', 'o3', 'o4']
    #     return any(model.startswith(f'{m}-') or model == m for m in reasoning_models)
    
    def _prepare_request_params(self, request: ChatRequest) -> Dict[str, Any]:
        """准备请求参数，包含思考功能配置"""
        
        # 转换消息格式
        if hasattr(request.messages, 'to_native'):
            messages = request.messages.to_native()
        elif isinstance(request.messages, list):
            messages = [msg.to_native() if hasattr(msg, 'to_native') else msg for msg in request.messages]
        else:
            raise ValueError("Invalid messages format")
        
        params = {
            "model": request.model,
            "messages": messages,
            "stream": request.stream,
        }
        
        # 检查是否为推理模型
        is_reasoning_model = request.include_thinking
        
        if is_reasoning_model:
            # 对于推理模型，使用max_completion_tokens而不是max_tokens
            if request.max_tokens:
                params["max_completion_tokens"] = request.max_tokens
        else:
            # 对于非推理模型，使用标准参数
            if request.max_tokens:
                params["max_tokens"] = request.max_tokens
        
        # 添加其他可选参数（推理模型不支持某些参数）
        if is_reasoning_model:
            # 推理模型支持的参数
            if request.temperature is not None:
                params["temperature"] = request.temperature
            
            # 推理努力程度（仅o1系列支持）
            if request.reasoning_effort:
                params["reasoning_effort"] = request.reasoning_effort
        else:
            # 标准模型支持的参数
            optional_params = ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty']
            for param in optional_params:
                value = getattr(request, param, None)
                if value is not None:
                    params[param] = value
        
        return params
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """发送聊天请求"""
        async with self.ensure_initialized():
            errors = self.validate_request(request)
            if errors:
                raise ValidationError(f"Request validation failed: {', '.join(errors)}", self.platform_type)
            
            start_time = time.time()
            
            try:
                if request.include_thinking:
                    # 使用流式方式处理思考功能
                    return await self._chat_with_thinking(request)
                else:
                    # 使用标准方式
                    params = self._prepare_request_params(request)
                    params["stream"] = False
                    
                    response = await self._client.chat.completions.create(**params)
                    processing_time = time.time() - start_time
                    
                    # 提取reasoning_tokens
                    thoughts = ""
                    if (response.usage and 
                        hasattr(response.usage, 'completion_tokens_details') and 
                        response.usage.completion_tokens_details and
                        hasattr(response.usage.completion_tokens_details, 'reasoning_tokens')):
                        reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
                        if reasoning_tokens and reasoning_tokens > 0:
                            thoughts = f"<reasoning_tokens>{reasoning_tokens}</reasoning_tokens>"
                    
                    content = response.choices[0].message.content or ""
                    if thoughts and request.include_thinking:
                        content = f"<answer>{content}</answer>\n"
                    
                    return ChatResponse(
                        content=content,
                        model=response.model,
                        usage=response.usage.model_dump() if response.usage else None,
                        finish_reason=response.choices[0].finish_reason,
                        processing_time=processing_time,
                        thoughts=thoughts if thoughts else None
                    )
                
            except Exception as e:
                self.logger.error(f"Chat request failed: {e}")
                raise self.convert_exception(e)
    
    async def _chat_with_thinking(self, request: ChatRequest) -> ChatResponse:
        """使用思考功能的聊天请求（通过流式处理模拟）"""
        params = self._prepare_request_params(request)
        params["stream"] = True
        
        thoughts = ""
        answer = ""
        
        try:
            stream = await self._client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # 对于OpenAI，reasoning信息在usage中，不在内容流中
                    answer += content
            
            # 获取reasoning信息
            if (hasattr(stream, 'usage') and stream.usage and 
                hasattr(stream.usage, 'completion_tokens_details') and 
                stream.usage.completion_tokens_details and
                hasattr(stream.usage.completion_tokens_details, 'reasoning_tokens')):
                reasoning_tokens = stream.usage.completion_tokens_details.reasoning_tokens
                if reasoning_tokens and reasoning_tokens > 0:
                    thoughts = f"<reasoning_tokens>{reasoning_tokens}</reasoning_tokens>"
            
            return ChatResponse(
                content=answer,
                model=request.model,
                finish_reason="stop",
                thoughts=thoughts if thoughts else ""
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI reasoning request failed: {e}")
            raise self.convert_exception(e)
    
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
                    params = self._prepare_request_params(request)
                    params["stream"] = True
                    
                    stream = await self._client.chat.completions.create(**params)
                    
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield StreamChunk(
                                content=chunk.choices[0].delta.content,
                                finish_reason=chunk.choices[0].finish_reason,
                                content_type="answer"
                            )
                            
            except Exception as e:
                self.logger.error(f"Stream request failed: {e}")
                raise self.convert_exception(e)
    
    async def _stream_with_thinking(self, request: ChatRequest) -> AsyncGenerator[StreamChunk, None]:
        """支持思考功能的流式处理"""
        params = self._prepare_request_params(request)
        params["stream"] = True
        
        thoughts_started = False
        answer_started = False
        reasoning_tokens = 0
        
        try:
            stream = await self._client.chat.completions.create(**params)
            
            async for chunk in stream:
                # OpenAI的o1系列模型不会在流中提供reasoning内容
                # reasoning信息在最终的usage中
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    
                    if not answer_started:
                        # 开始输出答案
                        yield StreamChunk(
                            content="<answer>" + content,
                            content_type="answer_start"
                        )
                        answer_started = True
                    else:
                        yield StreamChunk(
                            content=content,
                            content_type="answer"
                        )
                
                # 检查是否有reasoning信息
                if (chunk.usage and 
                    hasattr(chunk.usage, 'completion_tokens_details') and 
                    chunk.usage.completion_tokens_details and
                    hasattr(chunk.usage.completion_tokens_details, 'reasoning_tokens')):
                    reasoning_tokens = chunk.usage.completion_tokens_details.reasoning_tokens
            
            # 如果有reasoning tokens，输出思考信息
            if reasoning_tokens > 0 and not thoughts_started:
                yield StreamChunk(
                    content=f"<reasoning_tokens>{reasoning_tokens}</reasoning_tokens>\n",
                    content_type="thought"
                )
            
            # 标记结束
            yield StreamChunk(
                content="</answer>\n",
                finish_reason="stop",
                content_type="end"
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI reasoning stream failed: {e}")
            raise self.convert_exception(e)

    async def stream_chat_completion(self, request: ChatRequest) -> str:
        """流式聊天请求（返回完整内容）"""
        response = ""
        async for chunk in self.stream_chat(request):
            response += chunk.content
        return response
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

class GenaiProvider(Provider):
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
        if request.system_prompt:
            config.system_prompt = request.system_prompt
        # 设置思考功能
        if request.include_thinking or isinstance(request.thinking_budget, int):
            config.thinking_config = self._types.ThinkingConfig(
                thinking_budget=request.thinking_budget or -1,
                include_thoughts=True
            )
        
        return config
    
    def _message_to_native(self, messages: GeminiMessage | List[GeminiMessage]) -> List[Any]:
        if isinstance(messages, GeminiMessage):
            native_msg = messages.to_native()
            return [native_msg] if native_msg is not None else []
        elif isinstance(messages, list):
            native_messages = []
            for msg in messages:
                if msg is not None:
                    native_msg = msg.to_native()
                    if native_msg is not None:
                        native_messages.append(native_msg)
            return native_messages
        else:
            raise ValueError("Invalid messages format: %s" % messages)

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
                    contents = self._message_to_native(request.messages)
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
        contents = self._message_to_native(request.messages)
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
                    contents = self._message_to_native(request.messages)
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
        contents = self._message_to_native(request.messages)
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
        (PlatformType.GOOGLE_OFFICIAL, GenaiProvider),
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
    'GenaiProvider',
    'VLLMProvider',
]
