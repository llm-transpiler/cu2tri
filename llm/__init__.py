# -*- coding: utf-8 -*-
"""
大语言模型对话管理库
一个用于管理树形对话历史的库，旨在兼容多种大语言模型服务，并支持多模态内容。
"""

# 导入核心功能
from .providers import (
    # 基础类
    FileManager, Message, ChatHistory,
    Part, TextPart, ThoughtPart, ContentPart, FilePart, ImagePart,
)

from .history import (
    # 异常
    HistoryError, NodeNotFoundError, ProviderError, UnsupportedModalityError,
    
    # 节点和树
    MessageNode, ConversationTree,
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
] 