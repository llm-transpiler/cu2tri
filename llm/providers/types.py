# -*- coding: utf-8 -*-
"""
LLM提供商类型定义 - 三层架构
平台 → 厂商 → 模型的优雅分级设计
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Set
import re

# ============ 第一层：平台类型 ============

class PlatformCategory(str, Enum):
    """平台类别"""
    AGGREGATOR = "aggregator"       # 聚合平台 - 支持多厂商模型
    OFFICIAL = "official"           # 官方平台 - 厂商自有API
    LOCAL = "local"                 # 本地平台 - 本地推理服务
    CUSTOM = "custom"               # 自定义平台 - 用户自定义
    
    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

class PlatformType(str, Enum):
    """部署平台/公司, 付费入口"""
    # 聚合平台, 没有自研模型, 转发模型厂商提供的API, 或可能提供其他家的部署
    OPENROUTER = "openrouter"
    # LITELLM = "litellm"
    # TOGETHER = "together"
    # REPLICATE = "replicate"
    # HUGGINGFACE = "huggingface"
    # AZURE_OPENAI = "azure_openai"
    # SILICONFLOW = "siliconflow"
    
    # 提供API的公司, 具有自己的部署
    OPENAI_OFFICIAL = "openai"
    GOOGLE_OFFICIAL = "google"
    ANTHROPIC_OFFICIAL = "anthropic"
    DEEPSEEK_OFFICIAL = "deepseek"
    ZHIPU_OFFICIAL = "zhipu"
    # COHERE_OFFICIAL = "cohere"
    # MISTRAL_OFFICIAL = "mistral"
    
    # # 本地平台
    VLLM = "vllm"
    # OLLAMA = "ollama"
    # TEXT_GENERATION_WEBUI = "text_generation_webui"
    # LLAMACPP = "llamacpp"
    # TRANSFORMERS = "transformers"
    # FASTCHAT = "fastchat"
    # LMDEPLOY = "lmdeploy"
    
    # # 自定义平台
    # CUSTOM_HTTP = "custom_http"
    # CUSTOM_GRPC = "custom_grpc"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

class SDKType(str, Enum):
    """SDK类型, 只考虑python的SDK"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GENAI = "genai"
    
    VLLM = "vllm"
    
    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

# ============ 第二层：厂商类型 ============

class VendorType(str, Enum):
    """模型厂商名称, 用于openrouter中的{vendor_type}/{model_name}及类似场景"""
    # 商业厂商
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CLAUDE = "anthropic"
    GOOGLE = "google"
    # COHERE = "cohere"
    # MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    ZHIPU = "zhipu"             # 智谱AI
    # MOONSHOT = "moonshot"     # 月之暗面
    # BAICHUAN = "baichuan"     # 百川智能
    # ALIBABA = "alibaba"       # 阿里巴巴（通义千问）
    # BAIDU = "baidu"           # 百度（文心一言）
    # TENCENT = "tencent"       # 腾讯（混元）
    
    # # 开源厂商/组织
    # META = "meta"                 # Meta (Llama系列)
    # MICROSOFT = "microsoft"       # Microsoft (Phi系列)
    # NVIDIA = "nvidia"             # NVIDIA (Nemotron系列)
    # DATABRICKS = "databricks"     # Databricks (DBRX)
    # APPLE = "apple"               # Apple (OpenELM)
    # IBM = "ibm"                   # IBM (Granite)
    # SALESFORCE = "salesforce"     # Salesforce (CodeGen)
    # BIGSCIENCE = "bigscience"     # BigScience (BLOOM)
    # HUGGINGFACE = "huggingface"   # HuggingFace自有模型
    
    # # 开源社区
    # ALPACA = "alpaca"         # Alpaca系列
    # VICUNA = "vicuna"         # Vicuna系列
    # WIZARD = "wizard"         # WizardLM系列
    # ORCA = "orca"             # Orca系列
    # OPEN_CHAT = "open_chat"   # OpenChat系列
    
    # 聚合API自行部署 - 使用不同的value但可以有相同的显示名称
    OPENROUTER_DEEPSEEK = "openrouter_deepseek"
    
    def __str__(self) -> str:
        # 对于OpenRouter平台的DeepSeek，显示时仍然使用 "deepseek"
        if self == VendorType.OPENROUTER_DEEPSEEK:
            return "deepseek"
        return self.value

    def __repr__(self) -> str:
        return self.value
    
    @property 
    def display_name(self) -> str:
        """获取显示名称，用于UI展示"""
        if self == VendorType.OPENROUTER_DEEPSEEK:
            return "deepseek"
        return self.value

# ============ 第三层：模型定义 ============

@dataclass
class ModelSpec:
    """模型规格定义"""
    name: str                                       # 模型名称
    vendor: VendorType                              # 所属厂商
    size: Optional[str] = None                      # 模型大小 (7B, 13B, 70B等)
    variant: Optional[str] = None                   # 变体 (chat, instruct, base等)
    context_length: Optional[int] = None            # 上下文长度
    input_token_limit: Optional[int] = None         # 输入token限制
    output_token_limit: Optional[int] = None        # 输出token限制
    is_thinking: bool = False                       # 是否支持思考
    supports_streaming_output: bool = False         # 是否支持流式输出
    supports_function_calling_input: bool = False   # 是否支持函数调用
    supports_picture_input: bool = False            # 是否支持图片
    supports_audio_input: bool = False              # 是否支持音频
    supports_video_input: bool = False              # 是否支持视频
    supports_pdf_input: bool = False                # 是否支持pdf
    supports_structured_output: bool = False        # 是否支持结构化输出
    supports_code: bool = False                     # 是否支持代码
    is_open_source: bool = False                    # 是否开源
    # https://ai.google.dev/gemini-api/docs/pricing?hl=zh-cn
    input_cost_per_million_tokens: Optional[float] = None # 输入价格/百万token
    output_cost_per_million_tokens: Optional[float] = None # 输出价格/百万token
    aliases: List[str] = field(default_factory=list)  # 别名列表


# ============ 平台支持矩阵 ============

@dataclass 
class PlatformInfo:
    """平台信息"""
    type_: PlatformType
    category: PlatformCategory
    name: str
    description: str
    requires_api_key: bool = True
    api_key_name: str = None
    # supports_streaming: bool = True
    # supports_multimodal: bool = False
    http_base_url: Optional[str] = None
    base_url: str = ""  # 新增的base_url属性
    supported_vendors: Set[VendorType] = field(default_factory=set)
    supported_models: Set[str] = field(default_factory=set)
    supported_sdks: Set[SDKType] = field(default_factory=set)  # 支持的SDK类型
    model_name_format: str = "{model}"  # 模型名称格式化模板
