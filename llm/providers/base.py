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

from .config import DEFAULT_TEMPERATURE
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
                self.content = '\n\n'.join(text_parts)
    
    def __str__(self) -> str:
        return f"Message(role={self.role}, content={self.content})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __dict__(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,    
        }
    
    # @abstractmethod
    # def to_native_format(self) -> Any:
    #     pass

# class ChatMessage(Message):
#     """默认消息实现类"""
#     def to_native_format(self) -> Dict[str, Any]:
#         """转换为标准格式"""
#         return {
#             "role": self.role,
#             "content": self.content,
#         }

class ChatHistory(ABC):
    """聊天历史基类（兼容性接口）"""
    def __init__(self, messages: List[Message], id_: Optional[str] = None):
        self.messages = messages
        self.id = id_ if id_ else str(uuid.uuid4())
    
    @abstractmethod
    def to_native(self) -> List[Any]:
        pass
    
    def simple_print(self) -> str:
        # messages = self.to_native()
        messages = self.messages
        result = ""
        for msg in messages:
            role = msg.role
            content = msg.content
            if isinstance(content, list) and content:
                content = content[0].content
            result += f"[{role.upper()}] {content}\n"
        return result

@dataclass
class ChatRequest:
    """聊天请求统一数据模型"""
    messages: List[Message]
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = DEFAULT_TEMPERATURE
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
    
    @abstractmethod
    async def stream_chat_completion(self, request: ChatRequest) -> str:
        """流式聊天请求, 但一次性返回所有内容"""
        pass
    
    # ============ 可选接口 ============
    def list_models(self) -> List[str]:
        """列出可用模型"""
        from .registry import get_provider_registry
        try:
            registry = get_provider_registry()
            return registry.get_models_by_platform(self.platform_type)
        except:
            return []

    async def list_models_async(self) -> List[str]:
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
                    messages=[Message(role="user", content="Hello")],
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
