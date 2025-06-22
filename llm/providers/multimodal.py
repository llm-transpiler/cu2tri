# -*- coding: utf-8 -*-
"""
多模态和文件处理模块
整合来自chat目录的多模态消息处理和文件管理功能
"""
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base import Message, ChatHistory, FileManager
from ..history.parts import TextPart, FilePart, ImagePart, ThoughtPart, ContentPart

# ============ Google Gemini 相关实现 ============

class GeminiFileManager(FileManager):
    """Gemini文件管理器，上传文件到Gemini服务"""
    
    def __init__(self, client_instance, logger: Optional[logging.Logger] = None):
        try:
            import google.genai as genai
            from google.genai import types
            self.genai = genai
            self.types = types
        except ImportError:
            raise ImportError("google-genai is required for GeminiFileManager")
        
        self.client = client_instance
        self._cache: Dict[str, Any] = {}
        self.logger = logger or logging.getLogger(__name__)
    
    def process(self, file_path: str) -> FilePart:
        """上传文件到Gemini服务并缓存结果"""
        if file_path in self._cache:
            uploaded_file = self._cache[file_path]
        else:
            self.logger.info(f"Uploading '{file_path}' to Gemini service...")
            uploaded_file = self.client.files.upload(file=file_path)
            self._cache[file_path] = uploaded_file
        
        # 根据mime_type决定返回ImagePart还是FilePart
        if 'image' in uploaded_file.mime_type:
            return ImagePart(content=uploaded_file.uri, mime_type=uploaded_file.mime_type)
        else:
            return FilePart(content=uploaded_file.uri, mime_type=uploaded_file.mime_type)

class GeminiMessage(Message):
    """适用于Google Gemini API的多模态消息"""
    
    def to_native(self) -> Optional[Any]:
        """将消息转换为Gemini API格式"""
        try:
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai is required for GeminiMessage")
            
        native_parts = []
        for part in self.parts:
            if isinstance(part, TextPart):
                native_parts.append(types.Part.from_text(text=str(part.content)))
            elif isinstance(part, FilePart):
                native_parts.append(types.Part.from_uri(file_uri=part.content, mime_type=part.mime_type))
        
        if not native_parts or self.role == 'system':
            return None
        
        api_role = 'model' if self.role == 'assistant' else 'user'
        return types.Content(role=api_role, parts=native_parts)

class GeminiChatHistory(ChatHistory):
    """适用于Google Gemini API的对话历史"""
    def to_native(self) -> Optional[Any]:
        messages = self.messages
        if isinstance(messages, Message):
            messages = [messages]
        return [msg.to_native() for msg in messages]

# ============ OpenAI 相关实现 ============

class OpenRouterMessage(Message):
    """适用于OpenAI API的多模态消息"""
    
    def to_native(self) -> Dict[str, Any]:
        """将消息转换为OpenAI API格式"""
        from ..history.errors import UnsupportedModalityError
        
        native_parts = []
        text_content = ""
        
        for part in self.parts:
            if isinstance(part, ThoughtPart):
                text_content += f"<thought>\n{part.content}\n</thought>\n"
        for part in self.parts:
            if isinstance(part, TextPart):
                text_content += part.content
            elif isinstance(part, ImagePart):
                native_parts.append({"type": "image_url", "image_url": {"url": part.content}})
            elif isinstance(part, FilePart):
                raise UnsupportedModalityError("OpenAI Chat API does not directly support FileParts.")
        if text_content:
            native_parts.insert(0, {"type": "text", "text": text_content})
            return {"role": self.role, "content": native_parts}
        else:
            return {"role": self.role, "content": text_content}

class OpenRouterChatHistory(ChatHistory):
    """适用于OpenAI API的对话历史"""
    def to_native(self) -> Optional[Any]:
        messages = self.messages
        if isinstance(messages, Message):
            messages = [messages]
        return [msg.to_native() for msg in messages]

# ============ OpenRouter 相关实现 ============

class OpenRouterFileManager(FileManager):
    """OpenRouter文件管理器，将本地文件编码为base64格式"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def process(self, file_path: str) -> FilePart:
        """将本地文件编码为base64并返回适当的Part对象"""
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
            # 添加文件名属性
            setattr(file_part, 'filename', file_path_obj.name)
            return file_part
        else:
            # 默认处理为通用文件
            mime_type = 'application/octet-stream'
            data_url = f"data:{mime_type};base64,{file_content}"
            file_part = FilePart(content=data_url, mime_type=mime_type)
            setattr(file_part, 'filename', file_path_obj.name)
            return file_part

# class OpenAIMessage(OpenAIMessage):
#     """适用于OpenRouter API的多模态消息，完全兼容OpenAI SDK并扩展了PDF支持"""
#     pass

# class OpenAIChatHistory(ChatHistory):
#     """适用于OpenRouter API的对话历史"""
#     pass

# ============ 导出 ============

__all__ = [
    'GeminiFileManager',
    'GeminiMessage', 
    'GeminiChatHistory',
    'OpenRouterFileManager',
    'OpenRouterMessage',
    'OpenRouterChatHistory', 
    # 'OpenAIMessage',
    # 'OpenAIChatHistory',
    'create_file_manager',
    'create_message',
    'create_chat_history',
] 