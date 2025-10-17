# -*- coding: utf-8 -*-
"""
对话节点模块，定义对话树中的节点类
"""
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..providers.base import Message

class MessageNode:
    
    def __init__(self, message_obj: 'Message', parent: Optional['MessageNode'] = None):
        self.id = message_obj.id
        self.message = message_obj
        self.parent = parent
        self.children: list['MessageNode'] = []
        self.tree = None
        
        # 如果指定了父节点，建立双向关系
        if parent is not None:
            parent.add_child(self)
    
    def add_child(self, child_node: 'MessageNode'):
        if child_node not in self.children:
            self.children.append(child_node)
        
        if child_node.parent != self:
            child_node.parent = self
    
    def remove_child(self, child_node: 'MessageNode'):
        if child_node in self.children:
            self.children.remove(child_node)
        
        if child_node.parent == self:
            child_node.parent = None
    
    def remove_all_children(self):
        for child in self.children:
            child.parent = None
        self.children = []
    
    def get_all_children(self) -> list['MessageNode']:
        all_children = []
        for child in self.children:
            all_children.extend(child.get_all_children())
            all_children.append(child)
        return all_children

    def get_children(self) -> list['MessageNode']:
        return self.children
    
    def get_parent(self) -> Optional['MessageNode']:
        return self.parent
    
    def get_message(self) -> 'Message':
        return self.message
    
    def __repr__(self) -> str:
        return f"MessageNode(id={self.id}, role={self.message.role}, parts={len(self.message.parts)})" 