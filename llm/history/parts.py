# -*- coding: utf-8 -*-
"""
消息内容部分模块
定义消息中不同类型的内容部分 (Part)
"""
from abc import ABC
from typing import Optional

class Part(ABC):
    """消息内容部分的抽象基类"""
    def __init__(self, content: Optional[str] = None, mime_type: Optional[str] = None):
        self.content = content
        self.mime_type = mime_type

class TextPart(Part):
    """文本内容部分"""
    def __init__(self, content: Optional[str] = None):
        super().__init__(content, 'text/plain')

class ThoughtPart(TextPart):
    """思考过程部分，用于记录模型的内部思考"""
    pass

class ContentPart(TextPart):
    """正式回答内容部分"""
    pass

class FilePart(Part):
    """通用文件部分 (如 PDF)"""
    pass

class ImagePart(FilePart):
    """图像内容部分"""
    pass 