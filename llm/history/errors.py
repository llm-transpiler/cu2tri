# -*- coding: utf-8 -*-
"""
异常处理模块
定义对话历史管理相关的异常类
"""

class HistoryError(Exception):
    """历史记录相关的基础异常类"""
    pass

class NodeNotFoundError(HistoryError):
    """当指定的节点ID不存在时抛出的异常"""
    def __init__(self, node_id: str | None = None, tree_id: str | None = None):
        self.node_id = node_id
        self.tree_id = tree_id
        super().__init__(f"Node with ID '{node_id}' not found in the conversation tree: '{tree_id}'.")
    
    def __str__(self):
        if self.node_id is None:
            return "Node not found"
        return f"Node with ID '{self.node_id}' not found in the conversation tree: '{self.tree_id}'."

class ProviderError(HistoryError):
    """API提供商相关的异常"""
    pass

class UnsupportedModalityError(ProviderError):
    """不支持的模态类型异常"""
    pass 