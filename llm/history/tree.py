# -*- coding: utf-8 -*-
"""
对话树模块
定义对话树的基类和具体实现类
"""
import logging
import uuid
from typing import Dict, List, Optional, TYPE_CHECKING

from .node import MessageNode
from .errors import NodeNotFoundError, ProviderError
from .parts import Part, TextPart, ThoughtPart, ContentPart

if TYPE_CHECKING:
    from ..providers.base import Message, ChatHistory, FileManager

class ConversationTree:
    
    def __init__(self, message_cls: "Message" , history_cls: "ChatHistory", 
                 system_prompt: Optional[str] = "You are a helpful assistant.", file_manager: Optional["FileManager"] = None, 
                 logger: Optional[logging.Logger] = None):
        self.id = str(uuid.uuid4())  # 添加树的唯一标识符
        self.message_cls = message_cls
        self.history_cls = history_cls
        self.file_manager = file_manager
        self.root: Optional[MessageNode] = None
        
        if system_prompt:
            self.root = MessageNode(self.message_cls(role="system", parts=[TextPart(system_prompt)]))
        
        self.current_node = self.root
        self.nodes: Dict[str, MessageNode] = {self.root.id: self.root} if self.root else {}
        
        self.logger = logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(logging.StreamHandler())
            self.logger.propagate = False

    def _add_message(self, role: str, parts: List[Part]) -> MessageNode:
        message_obj = self.message_cls(role=role, parts=parts)
        parent_node = self.current_node
        if self.root is None:
            parent_node = None
        
        new_node = MessageNode(message_obj, parent=parent_node)
        if parent_node:
            parent_node.add_child(new_node)
        else:
            self.root = new_node
        
        self.current_node = new_node
        self.nodes[new_node.id] = new_node
        new_node.tree = self
        return new_node

    def add_message(self, role: str, text: Optional[str] = None, local_images: Optional[List[str]] = None, 
                    local_files: Optional[List[str]] = None, thought: Optional[str] = None, 
                    public_uris: Optional[List[Part]] = None) -> MessageNode:
        parts = []
        
        if public_uris:
            parts.extend(public_uris)

        if thought: 
            parts.append(ThoughtPart(thought))
        if text: 
            parts.append(ContentPart(text))
        
        if self.file_manager:
            for file_path in (local_images or []):
                parts.append(self.file_manager.process(file_path))
            for file_path in (local_files or []):
                parts.append(self.file_manager.process(file_path))
        elif local_images or local_files:
            raise ProviderError("A file_manager must be configured in ConversationTree to handle local files.")
        
        return self._add_message(role=role, parts=parts)

    def add_user_message(self, text: Optional[str] = None, local_images: Optional[List[str]] = None, 
                    local_files: Optional[List[str]] = None, thought: Optional[str] = None, 
                    public_uris: Optional[List[Part]] = None) -> MessageNode:
        return self.add_message("user", text, local_images, local_files, thought, public_uris)

    def add_assistant_message(self, text: Optional[str] = None, local_images: Optional[List[str]] = None, 
                    local_files: Optional[List[str]] = None, thought: Optional[str] = None, 
                    public_uris: Optional[List[Part]] = None) -> MessageNode:
        return self.add_message("assistant", text, local_images, local_files, thought, public_uris)
    
    def _get_node(self, node: str | MessageNode) -> MessageNode:
        if node is None: raise NodeNotFoundError(tree_id=self.id)
        if isinstance(node, str):
            if node not in self.nodes: raise NodeNotFoundError(node_id=node, tree_id=self.id)
            node = self.nodes[node]
            if node.tree != self: raise ProviderError("Node <%s>:<%s> is not in the tree: <%s>" % (node.id, node.tree.id, self.id))
        elif isinstance(node, MessageNode):
            if node.tree != self: raise ProviderError("Node <%s>:<%s> is not in the tree: <%s>" % (node.id, node.tree.id, self.id))
            if node.id not in self.nodes: raise NodeNotFoundError(node_id=node.id, tree_id=self.id)
        else:
            raise ProviderError("Invalid node type: <%s>" % type(node))
        return node
    
    def remove_node_from_tree_with_children(self, node: str | MessageNode) -> MessageNode:
        node = self._get_node(node)
        node.parent.remove_child(node)
        return node
    
    def remove_node_from_tree_with_concrete_children(self, node: str | MessageNode) -> MessageNode:
        node = self._get_node(node)
        for child in node.children:
            node.parent.add_child(child)
        self.remove_node_from_tree_with_children(node)
        node.remove_all_children()
        return node

    def switch_to_node(self, node: str | MessageNode) -> MessageNode:
        self.current_node = self._get_node(node)
        return self.current_node

    def get_history_from_node(self, node: str | MessageNode, max_depth: int = 10000) -> "ChatHistory":
        node = self._get_node(node)
        message_objects: List["Message"] = []
        while node and max_depth > 0:
            max_depth -= 1
            message_objects.append(node.message)
            node = node.parent
        message_objects.reverse()
        return self.history_cls(messages=message_objects)

    def get_history(self, max_depth: int = 10000) -> "ChatHistory":
        return self.get_history_from_node(self.current_node, max_depth)
    
    def pretty_print(self, node: Optional[MessageNode] = None, prefix: str = "", is_last: bool = True):
        if node is None:
            node = self.root
        if not node:
            print("[WARNING] Conversation tree is empty.")
            return
        
        parts_summary = []
        for part in node.message.parts:
            part_type = type(part).__name__
            content_summary = str(part.content)[:40] + '...' if len(str(part.content)) > 40 else str(part.content)
            parts_summary.append(f"{part_type}: '{content_summary}'")
        
        print(prefix + ("└── " if is_last else "├── ") + 
              f"[{node.message.role.upper()}] (id: {node.id}) Parts: " + ", ".join(parts_summary))
        
        children = node.children
        for i, child in enumerate(children):
            new_prefix = prefix + ("    " if is_last else "│   ")
            self.pretty_print(child, new_prefix, is_last=(i == len(children) - 1))

