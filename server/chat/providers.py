# -*- coding: utf-8 -*-
"""
聊天服务API提供商适配器
统一不同API提供商的接口
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator, List
import openai
import time

try:
    import google.genai as genai
    from google.genai import types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    genai = None
    types = None

from .deprecated.models import ChatRequest, ChatResponse, ErrorResponse, ModelConfig, ModelProvider, ModelName, get_provider_type_by_model_name

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
                usage=response.usage.model_dump() if response.usage else None,
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

class GeminiProvider(ChatProvider):
    """Google Gemini API提供商（使用新API方式的流式输出捕获）"""
    
    def __init__(self, config: ModelConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        
        if not GOOGLE_GENAI_AVAILABLE:
            raise ImportError("google-genai is required for GeminiProvider")
        
        # 创建客户端实例
        self.client = genai.Client(
            api_key=config.api_key,
            # http_options=types.HttpOptions(api_version='v1alpha')
        )
    
    def _prepare_contents(self, request: ChatRequest, history: list) -> str:
        """准备Gemini API格式的内容（简化为字符串）"""
        messages = []
        
        # 添加历史消息
        for msg in history:
            content = ""
            if hasattr(msg, 'thought') and msg.thought:
                content += f"<Thought>\n{msg.thought}\n</Thought>\n"
            content += msg.content
            
            role = "model" if msg.role == "assistant" else msg.role
            messages.append(f"{role}: {content}")
        
        # 添加当前请求
        content = ""
        if request.thought:
            content += f"<Thought>\n{request.thought}\n</Thought>\n"
        content += request.message
        
        messages.append(f"User: {content}")
        
        return "\n\n".join(messages)
    
    def _get_generate_config(self, request: ChatRequest) -> types.GenerateContentConfig: # type: ignore
        """获取生成配置，包含思考功能"""
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True
            )
        )
    
    async def chat(self, request: ChatRequest, history: list) -> ChatResponse:
        """发送聊天请求 - 使用新API方式的流式输出捕获"""
        start_time = time.time()
        
        try:
            contents = self._prepare_contents(request, history)
            config = self._get_generate_config(request)
            
            # 使用新API方式进行流式输出捕获
            thoughts = ""
            answer = ""
            
            for chunk in self.client.models.generate_content_stream(
                model=request.model.value,
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
            
            # 组装最终响应
            full_content = answer  # 主要内容为答案部分
            thought_content = thoughts if thoughts else None
            
            processing_time = time.time() - start_time
            
            return ChatResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                message=full_content,
                thought=thought_content,
                model=request.model.value,
                usage=None,  # Gemini API可能不返回usage信息
                finish_reason="stop",
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            return ErrorResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                error=str(e),
                error_type=type(e).__name__
            )
    
    async def stream_chat(self, request: ChatRequest, history: list) -> AsyncGenerator[str, None]:
        """流式聊天请求 - 使用新API方式"""
        try:
            contents = self._prepare_contents(request, history)
            config = self._get_generate_config(request)
            
            thoughts_yielded = False
            answer_yielded = False
            
            for chunk in self.client.models.generate_content_stream(
                model=request.model.value,
                contents=contents,
                config=config,
            ):
                for part in chunk.candidates[0].content.parts:
                    if not part.text:
                        continue
                    elif part.thought:
                        # 输出思考过程（可选，带标记）
                        if not thoughts_yielded:
                            yield "<Thought>\n"
                            thoughts_yielded = True
                        yield part.text
                    else:
                        # 输出答案内容
                        if thoughts_yielded and not answer_yielded:
                            yield "\n</Thought>\n\n"
                            answer_yielded = True
                        yield part.text
            
            # 如果有思考内容但没有输出结束标记
            if thoughts_yielded and not answer_yielded:
                yield "\n</Thought>\n"
                    
        except Exception as e:
            self.logger.error(f"Gemini streaming error: {e}")
            yield f"Error: {str(e)}"

class ProviderFactory:
    """提供商工厂类"""
    
    @staticmethod
    def create_provider(config: ModelConfig, logger: Optional[logging.Logger] = None) -> ChatProvider:
        """创建提供商实例"""
        provider_type = get_provider_type_by_model_name(config.model_name)
        
        if provider_type == "google_genai":
            if not GOOGLE_GENAI_AVAILABLE:
                raise ImportError(
                    "google-genai is required for Google genai models. "
                    "Please install it with: pip install google-genai"
                )
            return GeminiProvider(config, logger)
        
        elif provider_type == "openrouter":
            return OpenRouterProvider(config, logger)
        
        else:
            # 默认使用OpenRouter提供商（向后兼容）
            return OpenRouterProvider(config, logger)
    
    @staticmethod
    def list_available_models() -> Dict[str, List[str]]:
        """列出可用的模型，按提供商分类"""
        models = {
            "openrouter": [],
            "google_genai": []
        }
        
        for model in ModelName:
            if model.name.startswith("OPENROUTER_"):
                models["openrouter"].append(model.value)
            elif model.name.startswith("GOOGLE_GENAI_"):
                models["google_genai"].append(model.value)
            else:
                # 向后兼容的模型，根据值判断归类
                provider_type = get_provider_type_by_model_name(model.value)
                models[provider_type].append(model.value)
        
        return models 