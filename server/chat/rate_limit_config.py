# -*- coding: utf-8 -*-
"""
速率限制配置
基于OpenRouter API的实际限制
"""
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class OpenRouterRateLimits:
    """OpenRouter速率限制配置"""
    
    # 免费模型限制
    FREE_MODEL_RPM = 20          # 每分钟20个请求
    FREE_MODEL_DAILY_BASE = 50   # 购买<10 credits: 每日50个请求
    FREE_MODEL_DAILY_PAID = 1000 # 购买≥10 credits: 每日1000个请求
    
    # 付费模型限制（估算值，实际可能更高）
    PAID_MODEL_RPM = 300         # 每分钟300个请求
    
    # Token限制（保守估算）
    MAX_TOKENS_PER_MINUTE = 100000    # 每分钟最大token数
    MAX_TOKENS_PER_REQUEST = 4096     # 单次请求最大token数
    
    # 安全缓冲
    RATE_LIMIT_BUFFER = 0.8      # 使用限制的80%作为缓冲
    
    @classmethod
    def get_config(cls, has_paid_credits: bool = True) -> Dict[str, Any]:
        """获取配置字典
        
        Args:
            has_paid_credits: 是否购买了10+credits
        """
        return {
            'free_model_rpm': cls.FREE_MODEL_RPM,
            'free_model_daily': cls.FREE_MODEL_DAILY_PAID if has_paid_credits else cls.FREE_MODEL_DAILY_BASE,
            'paid_model_rpm': cls.PAID_MODEL_RPM,
            'max_tokens_per_minute': cls.MAX_TOKENS_PER_MINUTE,
            'max_tokens_per_request': cls.MAX_TOKENS_PER_REQUEST,
            'rate_limit_buffer': cls.RATE_LIMIT_BUFFER
        }

class ModelClassifier:
    """模型分类器"""
    
    # 免费模型列表
    FREE_MODELS = {
        'deepseek/deepseek-chat-v3-0324:free',
        'deepseek/deepseek-r1-0528:free',
        'meta-llama/llama-3.2-3b-instruct:free',
        'meta-llama/llama-3.2-1b-instruct:free',
        'microsoft/phi-3-mini-128k-instruct:free',
        'microsoft/phi-3-medium-128k-instruct:free',
        'huggingfaceh4/zephyr-7b-beta:free',
        'openchat/openchat-7b:free',
        'gryphe/mythomist-7b:free',
        'undi95/toppy-m-7b:free',
        'openrouter/auto:free'
    }
    
    # 付费模型（主要的）
    PAID_MODELS = {
        'openai/gpt-4o',
        'openai/gpt-4o-mini',
        'openai/gpt-4-turbo',
        'openai/gpt-3.5-turbo',
        'anthropic/claude-3-opus',
        'anthropic/claude-3-sonnet',
        'anthropic/claude-3-haiku',
        'google/gemini-pro-1.5',
        'google/gemini-2.0-flash-001',
        'meta-llama/llama-3.1-405b-instruct',
        'meta-llama/llama-3.1-70b-instruct',
        'meta-llama/llama-3.1-8b-instruct'
    }
    
    @classmethod
    def is_free_model(cls, model: str) -> bool:
        """判断是否为免费模型"""
        return model.endswith(':free') or model in cls.FREE_MODELS
    
    @classmethod
    def get_model_type(cls, model: str) -> str:
        """获取模型类型"""
        return 'free' if cls.is_free_model(model) else 'paid'
    
    @classmethod
    def get_recommended_models(cls) -> Dict[str, list]:
        """获取推荐模型列表"""
        return {
            'free': [
                'deepseek/deepseek-r1-0528:free',
                'deepseek/deepseek-chat-v3-0324:free',
                'meta-llama/llama-3.2-3b-instruct:free'
            ],
            'paid': [
                'openai/gpt-4o-mini',
                'google/gemini-2.0-flash-001',
                'anthropic/claude-3-haiku'
            ]
        }

class TokenEstimator:
    """Token估算器"""
    
    # 估算常数
    CHARS_PER_TOKEN = 4          # 平均4个字符=1个token
    SYSTEM_MESSAGE_TOKENS = 100  # 系统消息token数
    RESPONSE_FACTOR = 0.5        # 响应长度因子
    
    @classmethod
    def estimate_tokens(cls, message: str, thought: str = None, max_tokens: int = 1000) -> int:
        """估算请求token数"""
        # 输入文本token数
        message_tokens = len(message) // cls.CHARS_PER_TOKEN
        thought_tokens = len(thought or "") // cls.CHARS_PER_TOKEN
        
        # 总输入token数
        input_tokens = message_tokens + thought_tokens + cls.SYSTEM_MESSAGE_TOKENS
        
        # 预期输出token数
        output_tokens = int(max_tokens * cls.RESPONSE_FACTOR)
        
        # 总token数
        total_tokens = input_tokens + output_tokens
        
        return total_tokens
    
    @classmethod
    def estimate_cost(cls, tokens: int, model: str) -> float:
        """估算成本（美元）"""
        # 简化的成本估算
        if ModelClassifier.is_free_model(model):
            return 0.0
        
        # 付费模型的大致成本（每1K token）
        cost_per_1k = {
            'openai/gpt-4o': 0.005,
            'openai/gpt-4o-mini': 0.0001,
            'anthropic/claude-3-haiku': 0.0005,
            'google/gemini-2.0-flash-001': 0.0002
        }
        
        base_cost = cost_per_1k.get(model, 0.001)  # 默认每1K token $0.001
        return (tokens / 1000) * base_cost

# 环境变量配置
def get_env_config() -> Dict[str, Any]:
    """从环境变量获取配置"""
    return {
        'openrouter_api_key': os.getenv('OPENROUTER_API_KEY'),
        'has_paid_credits': os.getenv('OPENROUTER_PAID_CREDITS', 'true').lower() == 'true',
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'log_dir': os.getenv('QUEUE_LOG_DIR', 'logs/queue'),
        'max_workers': int(os.getenv('QUEUE_MAX_WORKERS', '10')),
        'max_queue_size': int(os.getenv('QUEUE_MAX_SIZE', '1000'))
    }

# 预设配置
DEVELOPMENT_CONFIG = {
    'max_workers': 5,
    'max_queue_size': 100,
    'task_timeout': 60,
    'rate_limit_buffer': 0.7,  # 更保守的限制
    'log_level': 'DEBUG'
}

PRODUCTION_CONFIG = {
    'max_workers': 20,
    'max_queue_size': 2000,
    'task_timeout': 300,
    'rate_limit_buffer': 0.8,
    'log_level': 'INFO'
}

TESTING_CONFIG = {
    'max_workers': 2,
    'max_queue_size': 10,
    'task_timeout': 30,
    'rate_limit_buffer': 0.5,  # 非常保守
    'log_level': 'DEBUG'
} 