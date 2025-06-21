# -*- coding: utf-8 -*-
"""
对话历史管理模块
提供树形结构的对话历史管理功能
"""

# 异常类
from .errors import HistoryError, NodeNotFoundError, ProviderError, UnsupportedModalityError

# 消息部分类
from .parts import Part, TextPart, ThoughtPart, ContentPart, FilePart, ImagePart

# 节点类
from .node import MessageNode

# 对话树类
from .tree import ConversationTree

__all__ = [
    # 异常
    'HistoryError', 'NodeNotFoundError', 'ProviderError', 'UnsupportedModalityError',
    
    # 消息部分
    'Part', 'TextPart', 'ThoughtPart', 'ContentPart', 'FilePart', 'ImagePart',
    
    # 节点
    'MessageNode',
    
    # 对话树
    'ConversationTree',
] 