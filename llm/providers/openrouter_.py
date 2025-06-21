# -*- coding: utf-8 -*-
"""
OpenRouter API 提供商模块
实现 OpenRouter API 的消息格式、历史记录处理和文件管理
应该暂时用不到pdf和image功能，先不做深入测试了
"""
# https://openrouter.ai/docs/features/images-and-pdfs
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .base import Message, ChatHistory, FileManager
from .parts import ThoughtPart, ContentPart, ImagePart, FilePart, Part

class OpenRouterMessage(Message):
    """适用于OpenRouter API的多模态消息，完全兼容OpenAI SDK并扩展了PDF支持"""
    def to_native_format(self) -> Dict[str, Any]:
        """将消息转换为OpenRouter API格式
        
        Returns:
            Dict[str, Any]: OpenRouter API消息格式，兼容OpenAI SDK
        """
        native_parts = []
        text_content = []
        
        for part in self.parts:
            if isinstance(part, ThoughtPart):
                text_content.append(f"<Thought>\n{part.content}\n</Thought>")
            elif isinstance(part, ContentPart):
                # 处理None值
                content = str(part.content) if part.content is not None else ""
                text_content.append(content)
            elif isinstance(part, ImagePart):
                # OpenRouter 支持标准的 image_url 格式
                native_parts.append({"type": "image_url", "image_url": {"url": part.content}})
            elif isinstance(part, FilePart):
                # OpenRouter 扩展：支持 PDF 文件
                filename = getattr(part, 'filename', 'document.pdf')
                native_parts.append({
                    "type": "file",
                    "file": {
                        "filename": filename,
                        "file_data": part.content
                    }
                })
        
        if text_content:
            native_parts.insert(0, {"type": "text", "text": "\n".join(text_content)})
        
        return {"role": self.role, "content": native_parts}

class OpenRouterChatHistory(ChatHistory):
    """适用于OpenRouter API的对话历史"""
    pass

class OpenRouterFileManager(FileManager):
    """OpenRouter文件管理器，将本地文件编码为base64格式"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(logging.StreamHandler())
            self.logger.propagate = False
    def process(self, file_path: str) -> Part:
        """将本地文件编码为base64并返回适当的Part对象
        
        Args:
            file_path: 本地文件路径
            
        Returns:
            Part: 编码后的Part对象（ImagePart或FilePart）
            
        Raises:
            FileNotFoundError: 当文件不存在时
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.logger.info(f"Encoding '{file_path}' to base64...")
        
        # 读取文件并编码
        with open(file_path, "rb") as file:
            file_content = base64.b64encode(file.read()).decode('utf-8')
        
        # 确定MIME类型和返回对应的Part对象
        suffix = file_path_obj.suffix.lower()
        if suffix in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
            data_url = f"data:{mime_type};base64,{file_content}"
            return ImagePart(content=data_url, mime_type=mime_type)
        elif suffix == '.png':
            mime_type = 'image/png'
            data_url = f"data:{mime_type};base64,{file_content}"
            return ImagePart(content=data_url, mime_type=mime_type)
        elif suffix == '.webp':
            mime_type = 'image/webp'
            data_url = f"data:{mime_type};base64,{file_content}"
            return ImagePart(content=data_url, mime_type=mime_type)
        elif suffix == '.pdf':
            mime_type = 'application/pdf'
            data_url = f"data:{mime_type};base64,{file_content}"
            file_part = FilePart(content=data_url, mime_type=mime_type)
            # 添加文件名属性以便OpenRouterMessage使用
            file_part.filename = file_path_obj.name
            return file_part
        else:
            # 默认处理为通用文件
            mime_type = 'application/octet-stream'
            data_url = f"data:{mime_type};base64,{file_content}"
            file_part = FilePart(content=data_url, mime_type=mime_type)
            file_part.filename = file_path_obj.name
            return file_part 