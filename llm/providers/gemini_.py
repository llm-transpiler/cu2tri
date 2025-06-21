# -*- coding: utf-8 -*-
"""
Google Gemini API 提供商模块
实现 Gemini API 的消息格式、历史记录处理和文件管理
"""
import logging
from typing import Dict, Any, Optional

try:
    from google.genai import types, client
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    types = None
    client = None

from .base import Message, ChatHistory, FileManager
from .parts import TextPart, FilePart, ImagePart

class GeminiFileManager(FileManager):
    """Gemini文件管理器，上传文件到Gemini服务"""
    
    def __init__(self, client_instance, logger: Optional[logging.Logger] = None):
        if not GOOGLE_GENAI_AVAILABLE:
            raise ImportError("google-genai is required for GeminiFileManager")
        
        self.client = client_instance
        self._cache: Dict[str, Any] = {}
        self.logger = logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(logging.StreamHandler())
            self.logger.propagate = False
    
    def process(self, file_path: str) -> FilePart:
        """上传文件到Gemini服务并缓存结果
        
        Args:
            file_path: 本地文件路径
            
        Returns:
            FilePart: 包含Gemini文件URI的Part对象
        """
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
    
    def to_native_format(self) -> Optional[Any]:
        """将消息转换为Gemini API格式
        
        Returns:
            Optional[types.Content]: Gemini API内容格式，system消息返回None
        """
        if not GOOGLE_GENAI_AVAILABLE:
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
    pass