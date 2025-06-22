# -*- coding: utf-8 -*-
"""
聊天服务数据模型
基于精简后的 /llm 模块的简化数据模型
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import time

class SupportedModel(str, Enum):
    """支持的模型枚举"""
    # OpenRouter 模型
    OPENROUTER_GPT4O = "openai/gpt-4o"
    OPENROUTER_GPT4O_MINI = "openai/gpt-4o-mini"
    OPENROUTER_CLAUDE_SONNET = "anthropic/claude-3.5-sonnet"
    OPENROUTER_DEEPSEEK_FREE = "deepseek/deepseek-r1-0528:free"
    OPENROUTER_GEMINI_PRO = "google/gemini-2.5-pro"
    
    # Google Official 模型
    GEMINI_PRO = "gemini-2.5-pro"
    GEMINI_FLASH = "gemini-2.5-flash"
    
    # OpenAI Official 模型
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"

class ChatRequest(BaseModel):
    """聊天请求模型"""
    conversation_id: str = Field(..., description="对话ID")
    message: str = Field(..., description="用户消息")
    model: SupportedModel = Field(default=SupportedModel.OPENROUTER_DEEPSEEK_FREE, description="使用的模型")
    
    # 可选参数
    system_prompt: Optional[str] = Field(None, description="系统提示")
    max_tokens: Optional[int] = Field(None, description="最大token数")
    temperature: Optional[float] = Field(None, description="温度参数")
    
    # 思考功能
    include_thinking: bool = Field(False, description="是否包含思考过程")
    thinking_budget: Optional[int] = Field(None, description="思考token预算")
    reasoning_effort: Optional[str] = Field(None, description="推理强度")
    
    # 流式输出
    stream: bool = Field(False, description="是否流式输出")
    
    # 自动生成
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="请求ID")
    timestamp: float = Field(default_factory=time.time, description="请求时间戳")

class ChatResponse(BaseModel):
    """聊天响应模型"""
    request_id: str = Field(..., description="请求ID")
    conversation_id: str = Field(..., description="对话ID")
    message: str = Field(..., description="AI回复消息")
    model: str = Field(..., description="使用的模型")
    
    # 可选字段
    thoughts: Optional[str] = Field(None, description="思考过程")
    usage: Optional[Dict[str, Any]] = Field(None, description="token使用情况")
    finish_reason: Optional[str] = Field(None, description="结束原因")
    processing_time: Optional[float] = Field(None, description="处理时间(秒)")
    
    # 自动生成
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="响应ID")
    timestamp: float = Field(default_factory=time.time, description="响应时间戳")

class ErrorResponse(BaseModel):
    """错误响应模型"""
    request_id: str = Field(..., description="请求ID")
    conversation_id: str = Field(..., description="对话ID")
    error: str = Field(..., description="错误信息")
    error_type: str = Field(..., description="错误类型")
    
    # 自动生成
    error_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="错误ID")
    timestamp: float = Field(default_factory=time.time, description="错误时间戳")

class ConversationMessage(BaseModel):
    """对话消息模型"""
    role: str = Field(..., description="角色 (user/assistant)")
    content: str = Field(..., description="消息内容")
    thoughts: Optional[str] = Field(None, description="思考过程")
    timestamp: float = Field(default_factory=time.time, description="消息时间戳")

class ConversationHistory(BaseModel):
    """对话历史模型"""
    conversation_id: str = Field(..., description="对话ID")
    messages: List[ConversationMessage] = Field(default_factory=list, description="消息列表")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    updated_at: float = Field(default_factory=time.time, description="更新时间")

class ServiceStats(BaseModel):
    """服务统计信息"""
    total_requests: int = Field(0, description="总请求数")
    successful_requests: int = Field(0, description="成功请求数")
    failed_requests: int = Field(0, description="失败请求数")
    average_response_time: float = Field(0.0, description="平均响应时间")
    active_conversations: int = Field(0, description="活跃对话数")
    uptime: float = Field(0.0, description="运行时间")
    
class HealthStatus(BaseModel):
    """健康状态"""
    status: str = Field("healthy", description="服务状态")
    timestamp: float = Field(default_factory=time.time, description="检查时间")
    version: str = Field("2.0.0", description="版本号")
    providers: Dict[str, bool] = Field(default_factory=dict, description="提供商状态") 