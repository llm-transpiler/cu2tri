# -*- coding: utf-8 -*-
"""
OpenAI API 提供商模块
实现 OpenAI API 的消息格式和历史记录处理
"""
from typing import Dict, Any, List

from .base import Message, ChatHistory
from .parts import ThoughtPart, ContentPart, ImagePart, FilePart
from ..history.errors import UnsupportedModalityError

class OpenAIMessage(Message):
    """适用于OpenAI API的多模态消息"""
    def to_native_format(self) -> Dict[str, Any]:
        """将消息转换为OpenAI API格式
        
        Returns:
            Dict[str, Any]: OpenAI API消息格式
            
        Raises:
            UnsupportedModalityError: 当消息包含OpenAI不支持的FilePart时
        """
        native_parts = []
        text_content = []
        
        for part in self.parts:
            if isinstance(part, ThoughtPart):
                text_content.append(f"<Thought>\n{part.content}\n</Thought>")
            elif isinstance(part, ContentPart):
                text_content.append(part.content)
            elif isinstance(part, ImagePart):
                native_parts.append({"type": "image_url", "image_url": {"url": part.content}})
            elif isinstance(part, FilePart):
                raise UnsupportedModalityError("OpenAI Chat API does not directly support FileParts.")
        
        if text_content:
            native_parts.insert(0, {"type": "text", "text": "\n".join(text_content)})
        
        return {"role": self.role, "content": native_parts}

class OpenAIChatHistory(ChatHistory):
    """适用于OpenAI API的对话历史"""
    pass 