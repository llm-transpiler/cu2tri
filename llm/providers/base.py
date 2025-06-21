# -*- coding: utf-8 -*-
"""
提供商基类模块
定义所有API提供商共用的抽象基类，支持三层架构（平台→厂商→模型）
"""
import uuid
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict, AsyncGenerator
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import asynccontextmanager

from .types import PlatformType, ModelSpec
from ..history.parts import Part, ContentPart, TextPart
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PlatformConfig
import time

# ============ 兼容性接口 ============

class FileManager(ABC):
    """文件处理管理器基类"""
    @abstractmethod
    def process(self, file_path: str) -> Part:
        pass

class Message(ABC):
    """消息基类（支持多模态）"""
    def __init__(self, role: str, parts: List[Part] = None, content: str = None, 
                 id_: Optional[str] = None, metadata: Dict[str, Any] = None, 
                 timestamp: Optional[str] = None):
        self.id = id_ if id_ else str(uuid.uuid4())
        self.role = role
        self.parts = parts or []
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 如果没有parts但有content，创建TextPart
        if not self.parts and self.content:
            self.parts = [ContentPart(content=self.content)]
        
        # 如果有parts但没有content，从parts中提取文本内容
        if self.parts and not self.content:
            
            text_parts = []
            for part in self.parts:
                if isinstance(part, TextPart) and part.content:
                    text_parts.append(str(part.content))
            if text_parts:
                self.content = '\n'.join(text_parts)
    
    @abstractmethod
    def to_native_format(self) -> Any:
        pass

class ChatMessage(Message):
    """默认消息实现类"""
    def to_native_format(self) -> Dict[str, Any]:
        """转换为标准格式"""
        return {
            "role": self.role,
            "content": self.content,
        }

class ChatHistory(ABC):
    """聊天历史基类（兼容性接口）"""
    def __init__(self, messages: List[Message], id_: Optional[str] = None):
        self.messages = messages
        self.id = id_ if id_ else str(uuid.uuid4())
    
    def to_native_format(self) -> List[Any]:
        return [message.to_native_format() for message in self.messages if message.to_native_format()]
    
    def simple_print(self) -> str:
        messages = self.to_native_format()
        result = ""
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '❌ No content')
            if isinstance(content, list) and content:
                content = content[0].get('text', str(content))
            result += f"[{role.upper()}] {content}\n"
        return result

# ============ 统一数据模型 ============

# @dataclass
# class ChatMessage:
#     """聊天消息统一数据模型"""
#     role: str  # user, assistant, system
#     content: str
#     metadata: Dict[str, Any] = field(default_factory=dict)
#     timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))
    
#     def __post_init__(self):
#         if not isinstance(self.metadata, dict):
#             self.metadata = {}

@dataclass
class ChatRequest:
    """聊天请求统一数据模型"""
    messages: List[ChatMessage]
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None # 控制模型重复使用相同词汇/短语的倾向, 设置为 0.5 可以让模型生成更多样化的词汇，避免啰嗦, 设置为 -0.5 可能让模型在某些上下文中保持一致的术语使用
    presence_penalty: Optional[float] = None # 控制模型引入新话题/概念的倾向
    stream: bool = False
    # 新增：思考功能配置
    include_thinking: bool = False  # 是否包含思考过程
    thinking_budget: Optional[int] = None  # OpenAI o1系列模型的思考token预算
    reasoning_effort: Optional[str] = None  # OpenAI o1系列模型的推理强度：'low', 'medium', 'high'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        if not self.messages:
            raise ValueError("Messages list cannot be empty")

@dataclass
class ChatResponse:
    """聊天响应统一数据模型"""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: Optional[float] = None
    # 新增：思考内容
    thoughts: Optional[str] = None
    
    def __post_init__(self):
        if not isinstance(self.metadata, dict):
            self.metadata = {}

@dataclass
class StreamChunk:
    """流式响应块统一数据模型"""
    content: str
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 新增：内容类型标识
    content_type: Optional[str] = None  # 'thought', 'answer', 'thought_start', 'thought_end', 'error', 'end'
    
    def __post_init__(self):
        if not isinstance(self.metadata, dict):
            self.metadata = {}
    
    def __str__(self) -> str:
        return self.content

# ============ 异常处理 ============

class ProviderError(Exception):
    """提供商错误基类"""
    def __init__(self, message: str, platform_type: Optional[PlatformType] = None, 
                 error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.platform_type = platform_type.value if platform_type else "unknown"
        self.error_code = error_code or ""
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "platform_type": self.platform_type,
            "error_code": self.error_code,
            "details": self.details
        }

class AuthenticationError(ProviderError):
    """认证错误"""
    pass

class RateLimitError(ProviderError):
    """限流错误"""
    pass

class ModelNotFoundError(ProviderError):
    """模型未找到错误"""
    pass

class ValidationError(ProviderError):
    """验证错误"""
    pass

class NetworkError(ProviderError):
    """网络错误"""
    pass

class ServiceUnavailableError(ProviderError):
    """服务不可用错误"""
    pass

# ============ 统一工具方法 ============

class RequestValidator:
    """请求验证器"""
    
    @staticmethod
    def validate_and_prepare_request(request: ChatRequest, config: 'PlatformConfig') -> List[str]:
        """验证并准备请求"""
        errors = []
        
        if not request.messages:
            errors.append("Messages list cannot be empty")
        
        # 设置默认值
        RequestValidator._set_default_values(request, config)
        
        # 验证参数范围
        errors.extend(RequestValidator._validate_parameters(request))
        
        return errors
    
    @staticmethod
    def _set_default_values(request: ChatRequest, config: 'PlatformConfig'):
        """设置默认值"""
        if request.model is None:
            preferred_models = getattr(config, 'preferred_models', [])
            if preferred_models:
                request.model = preferred_models[0]
        from .config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
        if request.max_tokens is None:
            request.max_tokens = getattr(config, 'max_tokens', DEFAULT_MAX_TOKENS)
        
        if request.temperature is None:
            request.temperature = getattr(config, 'temperature', DEFAULT_TEMPERATURE)
    
    @staticmethod
    def _validate_parameters(request: ChatRequest) -> List[str]:
        """验证参数范围"""
        errors = []
        
        if request.model is None:
            errors.append("Model name must be specified")
        
        if request.temperature is not None and not (0 <= request.temperature <= 2):
            errors.append("Temperature must be between 0 and 2")
        
        if request.max_tokens is not None and request.max_tokens <= 0:
            errors.append("Max tokens must be greater than 0")
        
        return errors

class ExceptionConverter:
    """异常转换器"""
    
    @staticmethod
    def convert_to_provider_error(e: Exception, platform_type: PlatformType) -> ProviderError:
        """将异常转换为统一的Provider错误"""
        error_str = str(e).lower()
        
        if "unauthorized" in error_str or "invalid api key" in error_str or "401" in error_str:
            return AuthenticationError(f"Authentication failed: {e}", platform_type)
        elif "rate limit" in error_str or "429" in error_str:
            return RateLimitError(f"Rate limit exceeded: {e}", platform_type)
        elif "not found" in error_str or "404" in error_str or "model" in error_str:
            return ModelNotFoundError(f"Model not found: {e}", platform_type)
        elif "timeout" in error_str or "network" in error_str:
            return NetworkError(f"Network error: {e}", platform_type)
        elif "503" in error_str or "service unavailable" in error_str:
            return ServiceUnavailableError(f"Service unavailable: {e}", platform_type)
        else:
            return ProviderError(f"Request failed: {e}", platform_type)

# ============ Provider基类设计 ============

class Provider(ABC):
    """统一的LLM提供商抽象基类"""
    
    def __init__(self, config: 'PlatformConfig', logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(f"{__name__}.{config.platform_type}")
        self._client = None
        self._initialized = False
        self._closed = False
    
    @property
    def platform_type(self) -> PlatformType:
        """获取平台类型"""
        return self.config.platform_type
    
    @property
    def name(self) -> str:
        """获取提供商名称"""
        return getattr(self.config, 'name', f"{self.platform_type}")
    
    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized and not self._closed
    
    @property
    def is_closed(self) -> bool:
        """检查是否已关闭"""
        return self._closed
    
    # ============ 生命周期管理 ============
    
    async def initialize(self) -> None:
        """初始化提供商（异步）"""
        if self._initialized and not self._closed:
            return
        
        if self._closed:
            raise ProviderError(f"Provider {self.name}:{self.platform_type} is closed, cannot be re-initialized")
        
        try:
            await self._initialize_client()
            self._initialized = True
            self._closed = False
            self.logger.info(f"Provider {self.name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Provider {self.name} initialization failed: {e}")
            raise ProviderError(f"Provider {self.name}:{self.platform_type} initialization failed: {e}") from e
    
    @abstractmethod
    async def _initialize_client(self) -> None:
        """初始化客户端（子类实现）"""
        pass
    
    async def close(self) -> None:
        """关闭连接和清理资源"""
        if self._closed:
            return
        
        try:
            await self._cleanup_client()
            self._closed = True
            self._initialized = False
            self.logger.info(f"Provider {self.name} closed successfully")
        except Exception as e:
            self.logger.warning(f"Provider {self.name} cleanup failed: {e}")
    
    async def _cleanup_client(self) -> None:
        """清理客户端资源（子类可重写）"""
        if hasattr(self, '_client') and self._client:
            if hasattr(self._client, 'close'):
                if asyncio.iscoroutinefunction(self._client.close):
                    await self._client.close()
                else:
                    self._client.close()
    
    @asynccontextmanager
    async def ensure_initialized(self):
        """确保初始化的上下文管理器"""
        if not self.is_initialized:
            await self.initialize()
        try:
            yield self
        finally:
            pass  # 保持连接，不自动关闭
    
    # ============ 核心接口 ============
    
    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """发送聊天请求"""
        pass
    
    @abstractmethod
    async def stream_chat(self, request: ChatRequest) -> AsyncGenerator[StreamChunk, None]:
        """流式聊天请求"""
        pass
    
    # ============ 可选接口 ============
    
    async def list_models(self) -> List[str]:
        """列出可用模型"""
        from .registry import get_provider_registry
        try:
            registry = get_provider_registry()
            return registry.get_models_by_platform(self.platform_type)
        except:
            return []
    
    async def get_model_info(self, model: str) -> Optional[ModelSpec]:
        """获取模型信息"""
        from .registry import get_provider_registry
        try:
            registry = get_provider_registry()
            return registry.get_model_spec(model)
        except:
            return None
    
    # ============ 验证和健康检查 ============
    
    def validate_request(self, request: ChatRequest) -> List[str]:
        """验证请求"""
        return RequestValidator.validate_and_prepare_request(request, self.config)
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with self.ensure_initialized():
                test_request = ChatRequest(
                    messages=[ChatMessage(role="user", content="Hello")],
                    max_tokens=1
                )
                await self.chat(test_request)
                return True
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False
    
    # ============ 工具方法 ============
    
    def convert_exception(self, e: Exception) -> ProviderError:
        """转换异常"""
        return ExceptionConverter.convert_to_provider_error(e, self.platform_type)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(platform_type='{self.platform_type}', name='{self.name}', initialized={self.is_initialized})"

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
    
    def _is_reasoning_model(self, model: str) -> bool:
        """检查是否为支持推理的模型"""
        if not model:
            return False
        
        reasoning_models = ['o1', 'o3', 'o4']
        return any(model.startswith(f'{m}-') or model == m for m in reasoning_models)
    
    def _prepare_request_params(self, request: ChatRequest) -> Dict[str, Any]:
        """准备请求参数，包含思考功能配置"""
        
        # 转换消息格式
        if hasattr(request.messages, 'to_native_format'):
            messages = request.messages.to_native_format()
        elif isinstance(request.messages, list):
            messages = [msg.to_native_format() if hasattr(msg, 'to_native_format') else msg for msg in request.messages]
        else:
            raise ValueError("Invalid messages format")
        
        params = {
            "model": request.model,
            "messages": messages,
            "stream": request.stream,
        }
        
        # 检查是否为推理模型
        is_reasoning_model = self._is_reasoning_model(request.model)
        
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
                if request.include_thinking and self._is_reasoning_model(request.model):
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
                if request.include_thinking and self._is_reasoning_model(request.model):
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