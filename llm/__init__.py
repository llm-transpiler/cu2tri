# -*- coding: utf-8 -*-
"""
大语言模型对话管理库
一个用于管理树形对话历史的库，旨在兼容多种大语言模型服务，并支持多模态内容。
"""

# 导入核心功能
from .providers import (
    # 基础类
    FileManager, Message, ChatHistory,
)

from .history.parts import (
    Part, TextPart, ThoughtPart, ContentPart, FilePart, ImagePart,
)

from .history import (
    # 异常
    HistoryError, NodeNotFoundError, ProviderError, UnsupportedModalityError,
    
    # 节点和树
    MessageNode, ConversationTree,
)

# OpenAI-compatible helpers (moved from cu2til/trans/dev_/llm.py)
from .client.openai_compat import (
    async_openai_llm_call,
    CallingIdentifier,
    get_api_params_method,
    get_api_param_openai_default,
    get_api_param_openai_openrouter,
    get_api_param_gemini_openai,
    get_api_param_deepseek_openai,
    get_api_param_anthropic_openrouter,
    make_openai_single_message,
    make_openai_message_system,
    make_openai_message_user,
    make_openai_message_assistant,
)

__version__ = "1.0.0"

__all__ = [
    # 基础类
    'FileManager', 'Message', 'ChatHistory',
    'Part', 'TextPart', 'ThoughtPart', 'ContentPart', 'FilePart', 'ImagePart',
    
    # 异常
    'HistoryError', 'NodeNotFoundError', 'ProviderError', 'UnsupportedModalityError',
    
    # 节点和树
    'MessageNode', 'ConversationTree',

    # OpenAI-compatible helpers
    'async_openai_llm_call',
    'CallingIdentifier', 'get_api_params_method',
    'get_api_param_openai_default', 'get_api_param_openai_openrouter',
    'get_api_param_gemini_openai', 'get_api_param_deepseek_openai',
    'get_api_param_anthropic_openrouter',
    'make_openai_single_message', 'make_openai_message_system',
    'make_openai_message_user', 'make_openai_message_assistant',
] 