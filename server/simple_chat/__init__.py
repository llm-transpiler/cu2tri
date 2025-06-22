# -*- coding: utf-8 -*-
"""
Chat Service v2.0
基于精简后的 /llm 模块的聊天服务
"""

# 导出核心组件
from .models import (
    ChatRequest, ChatResponse, ErrorResponse,
    ConversationHistory, ConversationMessage,
    ServiceStats, HealthStatus, SupportedModel
)

from .service import ChatService, ConversationManager

from .api_server import ChatAPIServer, create_chat_api_server

# 版本信息
__version__ = "2.0.0"
__all__ = [
    # 数据模型
    "ChatRequest", "ChatResponse", "ErrorResponse",
    "ConversationHistory", "ConversationMessage", 
    "ServiceStats", "HealthStatus", "SupportedModel",
    
    # 服务组件
    "ChatService", "ConversationManager",
    "ChatAPIServer", "create_chat_api_server",
] 