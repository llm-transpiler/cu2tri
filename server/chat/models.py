# -*- coding: utf-8 -*-
"""
聊天服务数据模型
定义请求、响应和配置模型
"""
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import time

class ModelProvider(str, Enum):
    """模型提供商枚举"""
    OPENAI = "openai"
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"

class ModelName(str, Enum):
    """支持的模型名称 - 基于OpenRouter实际支持的模型"""
    # OpenAI models (via OpenRouter)
    GPT_4O_MINI = "openai/gpt-4o-mini"
    GPT_4_1 = "openai/gpt-4.1"
    GPT_4_1_MINI = "openai/gpt-4.1-mini"
    O3_PRO = "openai/o3-pro"
    O3 = "openai/o3"
    O4_MINI_HIGH = "openai/o4-mini-high"
    CODEX_MINI = "openai/codex-mini"
    
    # Claude models (via OpenRouter)
    CLAUDE_OPUS_4 = "anthropic/claude-opus-4"
    CLAUDE_SONNET_4 = "anthropic/claude-sonnet-4"
    CLAUDE_3_7_SONNET = "anthropic/claude-3.7-sonnet"
    
    # DeepSeek models (via OpenRouter - 包含免费和付费模型)
    DEEPSEEK_CHAT_V3_FREE = "deepseek/deepseek-chat-v3-0324:free"
    DEEPSEEK_CHAT_V3 = "deepseek/deepseek-chat-v3-0324"
    DEEPSEEK_R1_FREE = "deepseek/deepseek-r1:free"
    DEEPSEEK_R1_0528_FREE = "deepseek/deepseek-r1-0528:free"
    DEEPSEEK_R1_QWEN3_FREE = "deepseek/deepseek-r1-0528-qwen3-8b:free"
    
    # Gemini models (via OpenRouter)
    GEMINI_2_5_PRO = "google/gemini-2.5-pro"
    GEMINI_2_5_PRO_PREVIEW = "google/gemini-2.5-pro-preview"
    GEMINI_2_5_PRO_PREVIEW_05_06 = "google/gemini-2.5-pro-preview-05-06"
    GEMINI_2_5_PRO_PREVIEW_06_05 = "google/gemini-2.5-pro-preview-06-05"
    GEMINI_2_0_FLASH = "google/gemini-2.0-flash"
    GEMINI_2_0_FLASH_001 = "google/gemini-2.0-flash-001"
    GEMINI_2_0_FLASH_LITE = "google/gemini-2.0-flash-lite"
    GEMINI_2_5_FLASH_PREVIEW_05_20 = "google/gemini-2.5-flash-preview-05-20"
    GEMINI_2_5_FLASH_PREVIEW_05_20_THINKING = "google/gemini-2.5-flash-preview-05-20:thinking"
    GEMINI_2_5_FLASH_PREVIEW_04_17 = "google/gemini-2.5-flash-preview-04-17"
    GEMINI_1_5_FLASH = "google/gemini-1.5-flash"
    GEMINI_1_5_FLASH_8B = "google/gemini-1.5-flash-8b"
    
    # Mistral models (via OpenRouter)
    MISTRAL_NEMO = "mistralai/mistral-nemo"
    
    # Meta models (via OpenRouter)
    LLAMA_3_3_70B_INSTRUCT = "meta-llama/llama-3.3-70b-instruct"

    # Gemini models (via genai)
    GENAI_GEMINI_2_5_PRO = "gemini-2.5-pro-preview-06-05"
    GENAI_GEMINI_2_0_FLASH = "gemini-2.0-flash"

@dataclass
class ChatMessage:
    """聊天消息"""
    role: Literal["user", "assistant", "system"]
    content: str
    thought: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class ChatRequest:
    """聊天请求"""
    conversation_id: str
    message: str
    model: ModelName
    thought: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

@dataclass
class ChatResponse:
    """聊天响应"""
    request_id: str
    conversation_id: str
    message: str
    model: str
    thought: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    processing_time: Optional[float] = None

@dataclass
class ErrorResponse:
    """错误响应"""
    request_id: str
    conversation_id: str
    error: str
    error_type: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class ModelConfig:
    """模型配置"""
    provider: ModelProvider
    model_name: str
    api_key: str
    base_url: Optional[str] = None
    max_tokens: int = 4096 * 4
    temperature: float = 0.7
    timeout: int = 30 