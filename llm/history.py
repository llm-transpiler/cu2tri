# -*- coding: utf-8 -*-
"""
一个用于管理树形对话历史的库，旨在兼容多种大语言模型服务，并支持多模态内容。
本库的设计哲学是将对话内容分解为不同的“部分”(Part)，并由专门的“提供商”(Provider)
实现将其转换为符合特定API要求的原生格式。
"""
import uuid
import base64
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type, Optional

# --------------------------------------------------------------------------
# Google GenAI types are used as a reference, but we avoid a hard dependency
# in the base classes. This allows the library to be used without google-genai
# if only other providers are needed.
try:
    from google.genai import types as GoogleGenAITypes
    from google import genai
except ImportError:
    GoogleGenAITypes = None
    GenerativeClient = None
# --------------------------------------------------------------------------


# ==============================================================================
# 模块: history.errors
# ==============================================================================
class HistoryError(Exception):
    """对话历史模块的基础异常类。"""
    pass

class NodeNotFoundError(HistoryError):
    """当在对话树中找不到指定的节点ID时抛出。"""
    def __init__(self, node_id: str):
        super().__init__(f"Node with ID '{node_id}' not found in the conversation tree.")

class ProviderError(HistoryError):
    """与模型提供商相关操作的错误。"""
    pass

class UnsupportedModalityError(ProviderError):
    """当提供商不支持某种内容模态时抛出。"""
    pass

# ==============================================================================
# 模块: providers.parts
# 说明: 定义消息中不同类型的内容部分 (Part)。
# ==============================================================================
class Part(ABC):
    """表示消息中单个内容部分的抽象基类。"""
    def __init__(self, content: Any):
        self.content = content

class TextPart(Part):
    """文本内容部分。"""
    pass

class ThoughtPart(TextPart):
    """模型的思考过程部分。"""
    pass

class UriPart(Part):
    """基于URI的内容部分，例如公开的图片URL。"""
    def __init__(self, content: str, mime_type: Optional[str] = None):
        super().__init__(content)
        self.mime_type = mime_type

class Base64Part(Part):
    """基于Base64编码的内容部分，用于本地文件。"""
    def __init__(self, content: str, mime_type: str, filename: str):
        super().__init__(content)
        self.mime_type = mime_type
        self.filename = filename

class AnnotationPart(Part):
    """用于存储和重用文件解析结果的注解部分。"""
    pass

# ==============================================================================
# 模块: providers.base
# ==============================================================================
class FileManager(ABC):
    """文件管理器的抽象基类。"""
    @abstractmethod
    def process_local_file(self, file_path: str) -> Part:
        """处理本地文件，返回一个适用于特定提供商的Part对象。"""
        pass

class Message(ABC):
    """消息对象的抽象基类。"""
    def __init__(self, role: str, parts: List[Part], id_: Optional[str] = None):
        self.id = id_ if id_ else str(uuid.uuid4())
        self.role = role
        self.parts = parts
    
    @abstractmethod
    def to_native_format(self) -> Any:
        """将消息转换为特定API提供商所需的原生格式。"""
        pass

class ChatHistory(ABC):
    """对话历史对象的抽象基类。"""
    def __init__(self, messages: List[Message], id_: Optional[str] = None):
        self.id = id_ if id_ else str(uuid.uuid4())
    
    def to_native_format(self) -> List[Any]:
        """将整个历史记录转换为原生格式列表。"""
        return [message.to_native_format() for message in self.messages if message.to_native_format()]

# ==============================================================================
# 模块: providers.openrouter
# ==============================================================================
class OpenRouterFileManager(FileManager):
    """适用于OpenRouter的文件管理器，将文件编码为Base64。"""
    def process_local_file(self, file_path: str) -> Base64Part:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode('utf-8')
        except FileNotFoundError:
            raise ProviderError(f"File not found at path: {file_path}")

        mime_type = 'application/pdf'
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            ext = filename.lower().split('.')[-1]
            ext = 'jpeg' if ext == 'jpg' else ext
            mime_type = f'image/{ext}'
            
        return Base64Part(content=encoded_string, mime_type=mime_type, filename=filename)

class OpenRouterMessage(Message):
    """适用于OpenRouter API的多模态消息。"""
    def to_native_format(self) -> Dict[str, Any]:
        native_content_parts = []
        text_parts = []
        annotations = None

        for part in self.parts:
            if isinstance(part, ThoughtPart):
                text_parts.append(f"<Thought>\n{part.content}\n</Thought>")
            elif isinstance(part, TextPart):
                text_parts.append(part.content)
            elif isinstance(part, UriPart):
                # OpenRouter 支持公开的图像URL
                native_content_parts.append({"type": "image_url", "image_url": {"url": part.content}})
            elif isinstance(part, Base64Part):
                data_url = f"data:{part.mime_type};base64,{part.content}"
                if 'image' in part.mime_type:
                    native_content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
                elif 'pdf' in part.mime_type:
                    # 根据文档，PDF也通过base64发送
                    native_content_parts.append({"type": "file", "file": {"filename": part.filename, "file_data": data_url}})
            elif isinstance(part, AnnotationPart):
                annotations = part.content
        
        if text_parts:
            native_content_parts.insert(0, {"type": "text", "text": "\n".join(text_parts)})
        
        message_dict = {"role": self.role, "content": native_content_parts}
        if annotations:
            message_dict["annotations"] = annotations
            
        return message_dict

class OpenRouterChatHistory(ChatHistory):
    """适用于OpenRouter API的对话历史。"""
    pass

# ==============================================================================
# 模块: history.node 和 history.tree
# ==============================================================================
class MessageNode:
    """表示对话树中的一个节点。"""
    def __init__(self, message_obj: Message, parent: Optional['MessageNode'] = None):
        self.id = message_obj.id
        self.message = message_obj
        self.parent = parent
        self.children: List['MessageNode'] = []
    def add_child(self, child_node: 'MessageNode'):
        self.children.append(child_node)
    def __repr__(self) -> str:
        return f"MessageNode(id={self.id}, role={self.message.role}, parts={len(self.message.parts)})"

class ConversationTree:
    """管理整个对话的树形结构。"""
    def __init__(self, message_cls: Type[Message], history_cls: Type[ChatHistory], 
                 system_prompt: Optional[str] = None, file_manager: Optional[FileManager] = None):
        self.message_cls = message_cls
        self.history_cls = history_cls
        self.file_manager = file_manager
        self.root: Optional[MessageNode] = None
        if system_prompt:
            self.root = MessageNode(self.message_cls(role="system", parts=[TextPart(system_prompt)]))
        self.current_node = self.root
        self.nodes: Dict[str, MessageNode] = {self.root.id: self.root} if self.root else {}

    def _add_message_from_parts(self, role: str, parts: List[Part]) -> MessageNode:
        """内部核心方法：使用Part列表添加一条新消息。"""
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
        return new_node

    def add_message(self, role: str, text: Optional[str] = None, local_files: Optional[List[str]] = None, 
                    public_uris: Optional[List[str]] = None, thought: Optional[str] = None, annotations: Optional[Any] = None) -> MessageNode:
        """
        通用方法，用于添加包含不同内容部分的消息。

        Args:
            role: 消息的角色 ('user' or 'assistant')。
            text: 文本内容。
            local_files: 需要由文件管理器处理的本地文件路径列表 (图像或PDF)。
            public_uris: 公开可访问的URI列表 (通常是图片)。
            thought: AI的思考过程。
            annotations: 从先前响应中获得的文件解析注解。

        Returns:
            新创建的消息节点。
        """
        parts: List[Part] = []
        if thought: parts.append(ThoughtPart(thought))
        if text: parts.append(TextPart(text))
        if annotations: parts.append(AnnotationPart(annotations))
        if public_uris:
            parts.extend([UriPart(uri) for uri in public_uris])
        if local_files:
            if not self.file_manager:
                raise ProviderError("A file_manager must be configured in ConversationTree to handle local files.")
            for file_path in local_files:
                parts.append(self.file_manager.process_local_file(file_path))
        return self._add_message_from_parts(role=role, parts=parts)
        
    def add_user_message(self, text: Optional[str] = None, local_files: Optional[List[str]] = None, 
                         public_uris: Optional[List[str]] = None) -> MessageNode:
        """添加一条用户消息的便捷方法。"""
        return self.add_message("user", text=text, local_files=local_files, public_uris=public_uris)

    def add_assistant_message(self, text: Optional[str] = None, thought: Optional[str] = None, 
                              annotations: Optional[Any] = None) -> MessageNode:
        """添加一条助手消息的便捷方法。"""
        return self.add_message("assistant", text=text, thought=thought, annotations=annotations)


    def switch_to_node(self, node_id: str):
        if node_id not in self.nodes:
            raise NodeNotFoundError(node_id)
        self.current_node = self.nodes[node_id]

    def get_history_for_api(self) -> ChatHistory:
        message_objects: List[Message] = []
        node = self.current_node
        while node:
            message_objects.append(node.message)
            node = node.parent
        message_objects.reverse()
        return self.history_cls(messages=message_objects)
    
    def pretty_print(self, node: Optional[MessageNode] = None, prefix: str = "", is_last: bool = True):
        """以美观、易读的格式打印整个对话树。"""
        if node is None:
            node = self.root
        if not node:
            print("[INFO] Conversation tree is empty.")
            return
        parts_summary = []
        for part in node.message.parts:
            part_type = type(part).__name__
            content_summary = str(part.content)[:40] + '...' if len(str(part.content)) > 40 else str(part.content)
            parts_summary.append(f"{part_type}: '{content_summary}'")
        print(prefix + ("└── " if is_last else "├── ") + f"[{node.message.role.upper()}] (id: {node.id}) Parts: " + ", ".join(parts_summary))
        children = node.children
        for i, child in enumerate(children):
            new_prefix = prefix + ("    " if is_last else "│   ")
            self.pretty_print(child, new_prefix, is_last=(i == len(children) - 1))

# ==============================================================================
# 模块: example.py
# ==============================================================================
if __name__ == "__main__":
    # --- 准备用于演示的临时文件 ---
    if not os.path.exists("temp_image.png"):
        with open("temp_image.png", "wb") as f:
            f.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="))
    if not os.path.exists("temp_doc.pdf"):
        with open("temp_doc.pdf", "wb") as f:
            f.write(b'%PDF-1.4\n1 0 obj\n<<>>\nendobj\n')

    # --- OpenRouter 示例 ---
    print("================== OpenRouter Example ==================")
    or_fm = OpenRouterFileManager()
    or_tree = ConversationTree(
        message_cls=OpenRouterMessage, 
        history_cls=OpenRouterChatHistory, 
        system_prompt="You are an expert analyst.",
        file_manager=or_fm
    )
    
    # 1. 使用便捷方法添加包含本地PDF和公开图片的用户消息
    or_tree.add_user_message(
        text="Please analyze this PDF and describe the cat in the picture.",
        local_files=["temp_doc.pdf"],
        public_uris=["https://placekitten.com/200/300"]
    )

    # 2. 使用便捷方法添加包含思考和注解的助手回复
    mock_annotations = [{"type": "source", "source": {"page": 1, "filename": "temp_doc.pdf"}}]
    or_tree.add_assistant_message(
        thought="The user provided a PDF and an image. I will first summarize the document based on my analysis, then describe the cat.",
        text="The document discusses advanced AI techniques. The image contains a cute kitten.",
        annotations=mock_annotations
    )

    print("\n--- OpenRouter Tree ---")
    or_tree.pretty_print()

    print("\n--- OpenRouter Native Format for API ---")
    native_history = or_tree.get_history_for_api().to_native_format()
    import json
    print(json.dumps(native_history, indent=2))

    # --- 清理临时文件 ---
    os.remove("temp_image.png")
    os.remove("temp_doc.pdf")
