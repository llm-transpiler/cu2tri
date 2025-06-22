# -*- coding: utf-8 -*-
"""
LLM交互处理器
负责与大语言模型交互，生成和优化内核代码
"""
import os
import sys
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import traceback

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from .base import BaseProcessor
from ..models import (
    LLMRequest, KernelResponse, LLMResult, 
    TaskType, TaskStatus
)

# 导入LLM相关库
try:
    from llm.chat.openrouter_ import OpenRouterProvider
    from llm.chat.gemini_ import GeminiChatProvider
    from llm.history.tree import ConversationTree
    from llm.history.node import ConversationNode
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

class LLMProcessor(BaseProcessor):
    """LLM交互处理器"""
    
    def __init__(self, logger=None):
        super().__init__(logger)
        
        # LLM提供商
        self.providers = {}
        self.conversations = {}  # 对话历史管理
        
        # 初始化LLM提供商
        self._initialize_providers()
        
        # 内核开发相关的提示模板
        self.prompt_templates = {
            "generate": self._get_generation_prompt_template(),
            "optimize": self._get_optimization_prompt_template(),
            "debug": self._get_debug_prompt_template(),
            "explain": self._get_explanation_prompt_template()
        }
    
    def _initialize_providers(self):
        """初始化LLM提供商"""
        if not LLM_AVAILABLE:
            self.logger.warning("LLM libraries not available")
            return
        
        try:
            # OpenRouter提供商
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                self.providers["openrouter"] = OpenRouterProvider(
                    api_key=openrouter_key,
                    logger=self.logger
                )
                self.logger.info("OpenRouter provider initialized")
            
            # Gemini提供商 - 使用流式输出捕获
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                self.providers["gemini"] = GeminiChatProvider(
                    api_key=gemini_key,
                    logger=self.logger
                )
                self.logger.info("Gemini provider initialized with stream capture")
            
            if not self.providers:
                self.logger.error("No LLM providers available - check API keys")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM providers: {e}")
    
    async def process(self, 
                     request: LLMRequest, 
                     gpu_ids: List[int], 
                     context: Dict[str, Any]) -> KernelResponse:
        """处理LLM交互请求"""
        start_time = time.time()
        self.log_task_start("LLM Generation", request.request_id)
        
        try:
            if not LLM_AVAILABLE or not self.providers:
                raise Exception("LLM services not available")
            
            # 执行LLM任务
            result = await self._process_llm_request(request, context)
            
            # 创建响应
            response = KernelResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                task_type=TaskType.LLM_GENERATION,
                status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
                result=result,
                processing_time=time.time() - start_time
            )
            
            self.log_task_end("LLM Generation", request.request_id, result.success, response.processing_time)
            return response
            
        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            self.logger.error(f"LLM processing failed for request {request.request_id}: {error_msg}")
            
            return self.create_error_response(
                request, TaskType.LLM_GENERATION, error_msg, 
                "LLMProcessingError", error_traceback
            )
    
    async def _process_llm_request(self, request: LLMRequest, context: Dict[str, Any]) -> LLMResult:
        """处理LLM请求"""
        start_time = time.time()
        
        try:
            # 获取对话历史
            conversation = self._get_or_create_conversation(request.conversation_id)
            
            # 构建提示
            prompt = self._build_prompt(request, context)
            
            # 选择提供商
            provider = self._select_provider(request.model_name)
            
            # 发送请求到LLM
            response = await self._call_llm(provider, prompt, request)
            
            # 解析响应
            parsed_result = self._parse_llm_response(response, request.task_type)
            
            # 更新对话历史
            self._update_conversation(conversation, prompt, response, request)
            
            return LLMResult(
                success=True,
                generated_code=parsed_result.get("code"),
                explanation=parsed_result.get("explanation"),
                suggestions=parsed_result.get("suggestions", []),
                model_used=request.model_name,
                tokens_used=response.get("usage", {}),
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return LLMResult(
                success=False,
                model_used=request.model_name,
                processing_time=time.time() - start_time
            )
    
    def _get_or_create_conversation(self, conversation_id: str):
        """获取或创建对话"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationTree(
                system_prompt=self._get_system_prompt(),
                logger=self.logger
            )
        return self.conversations[conversation_id]
    
    def _build_prompt(self, request: LLMRequest, context: Dict[str, Any]) -> str:
        """构建提示"""
        template = self.prompt_templates.get(request.task_type, self.prompt_templates["generate"])
        
        # 准备模板变量
        template_vars = {
            "user_message": request.user_message,
            "cuda_code": request.cuda_code or "",
            "triton_code": request.triton_code or "",
            "error_info": request.error_info or "",
            "performance_data": json.dumps(request.performance_data or {}, indent=2),
            "context": json.dumps(context, indent=2)
        }
        
        # 填充模板
        try:
            prompt = template.format(**template_vars)
        except KeyError as e:
            self.logger.warning(f"Missing template variable: {e}")
            prompt = f"{request.user_message}\n\nContext: {json.dumps(context, indent=2)}"
        
        return prompt
    
    def _select_provider(self, model_name: str):
        """选择合适的提供商"""
        if "gemini" in model_name.lower() and "gemini" in self.providers:
            return self.providers["gemini"]
        elif "openrouter" in self.providers:
            return self.providers["openrouter"]
        elif self.providers:
            return next(iter(self.providers.values()))
        else:
            raise Exception("No LLM provider available")
    
    async def _call_llm(self, provider, prompt: str, request: LLMRequest) -> Dict[str, Any]:
        """调用LLM提供商"""
        try:
            # 构建消息
            messages = [{"role": "user", "content": prompt}]
            
            # 调用提供商
            response = await provider.chat_completion(
                messages=messages,
                model=request.model_name,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            raise Exception(f"LLM API call failed: {e}")
    
    def _parse_llm_response(self, response: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            content = response.get("content", "")
            
            if task_type == "generate":
                return self._parse_generation_response(content)
            elif task_type == "optimize":
                return self._parse_optimization_response(content)
            elif task_type == "debug":
                return self._parse_debug_response(content)
            elif task_type == "explain":
                return self._parse_explanation_response(content)
            else:
                return {"explanation": content}
                
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return {"explanation": response.get("content", "")}
    
    def _parse_generation_response(self, content: str) -> Dict[str, Any]:
        """解析代码生成响应"""
        result = {"explanation": content}
        
        # 尝试提取代码块
        code_blocks = self._extract_code_blocks(content)
        if code_blocks:
            result["code"] = code_blocks[0]  # 使用第一个代码块
        
        # 尝试提取建议
        suggestions = self._extract_suggestions(content)
        if suggestions:
            result["suggestions"] = suggestions
        
        return result
    
    def _parse_optimization_response(self, content: str) -> Dict[str, Any]:
        """解析优化响应"""
        result = {"explanation": content}
        
        # 提取优化后的代码
        code_blocks = self._extract_code_blocks(content)
        if code_blocks:
            result["code"] = code_blocks[0]
        
        # 提取优化建议
        suggestions = self._extract_optimization_suggestions(content)
        if suggestions:
            result["suggestions"] = suggestions
        
        return result
    
    def _parse_debug_response(self, content: str) -> Dict[str, Any]:
        """解析调试响应"""
        result = {"explanation": content}
        
        # 提取修复后的代码
        code_blocks = self._extract_code_blocks(content)
        if code_blocks:
            result["code"] = code_blocks[0]
        
        # 提取调试建议
        suggestions = self._extract_debug_suggestions(content)
        if suggestions:
            result["suggestions"] = suggestions
        
        return result
    
    def _parse_explanation_response(self, content: str) -> Dict[str, Any]:
        """解析解释响应"""
        return {"explanation": content}
    
    def _extract_code_blocks(self, content: str) -> List[str]:
        """提取代码块"""
        import re
        
        # 匹配各种代码块格式
        patterns = [
            r'```(?:cuda|c\+\+|cpp|c)\n(.*?)```',  # CUDA/C++代码块
            r'```(?:python|py)\n(.*?)```',         # Python代码块
            r'```\n(.*?)```',                      # 通用代码块
            r'```(.*?)```'                         # 简单代码块
        ]
        
        code_blocks = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            code_blocks.extend(matches)
        
        return [block.strip() for block in code_blocks if block.strip()]
    
    def _extract_suggestions(self, content: str) -> List[str]:
        """提取建议"""
        suggestions = []
        
        # 查找建议相关的部分
        import re
        suggestion_patterns = [
            r'建议[：:]\s*(.+?)(?=\n\n|\n$|$)',
            r'Suggestion[：:]\s*(.+?)(?=\n\n|\n$|$)',
            r'优化[：:]\s*(.+?)(?=\n\n|\n$|$)',
            r'改进[：:]\s*(.+?)(?=\n\n|\n$|$)'
        ]
        
        for pattern in suggestion_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            suggestions.extend(matches)
        
        return [s.strip() for s in suggestions if s.strip()]
    
    def _extract_optimization_suggestions(self, content: str) -> List[str]:
        """提取优化建议"""
        return self._extract_suggestions(content)
    
    def _extract_debug_suggestions(self, content: str) -> List[str]:
        """提取调试建议"""
        return self._extract_suggestions(content)
    
    def _update_conversation(self, conversation, prompt: str, response: Dict[str, Any], request: LLMRequest):
        """更新对话历史"""
        try:
            # 添加用户消息
            conversation.add_message(
                role="user",
                text=prompt,
                thought=None
            )
            
            # 添加助手回复
            conversation.add_message(
                role="assistant", 
                text=response.get("content", ""),
                thought=None
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to update conversation: {e}")
    
    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        return """你是一个专业的CUDA和Triton内核开发专家。你的任务是帮助用户开发、优化和调试GPU内核。

你的专业技能包括：
1. CUDA编程和优化
2. Triton内核开发
3. GPU内存管理和性能优化
4. 并行计算算法设计
5. 内核调试和性能分析

请始终提供：
- 清晰的代码示例
- 详细的解释说明
- 实用的优化建议
- 潜在问题的警告

用中文回答用户问题。"""
    
    def _get_generation_prompt_template(self) -> str:
        """获取代码生成提示模板"""
        return """用户请求：{user_message}

上下文信息：
{context}

现有CUDA代码（如果有）：
```cuda
{cuda_code}
```

现有Triton代码（如果有）：
```python
{triton_code}
```

请根据用户的需求生成或修改内核代码。请提供：
1. 完整的代码实现
2. 详细的实现说明
3. 性能优化建议
4. 使用注意事项

请用代码块格式提供代码，并用中文进行说明。"""
    
    def _get_optimization_prompt_template(self) -> str:
        """获取优化提示模板"""
        return """用户请求：{user_message}

当前性能数据：
{performance_data}

需要优化的代码：
```cuda
{cuda_code}
```

```python
{triton_code}
```

上下文信息：
{context}

请分析当前代码的性能瓶颈，并提供优化建议。请包括：
1. 性能瓶颈分析
2. 优化后的代码
3. 优化原理说明
4. 预期性能提升

请用代码块格式提供优化后的代码。"""
    
    def _get_debug_prompt_template(self) -> str:
        """获取调试提示模板"""
        return """用户请求：{user_message}

错误信息：
{error_info}

问题代码：
```cuda
{cuda_code}
```

```python
{triton_code}
```

上下文信息：
{context}

请帮助调试和修复代码问题。请提供：
1. 错误原因分析
2. 修复后的代码
3. 修复说明
4. 避免类似问题的建议

请用代码块格式提供修复后的代码。"""
    
    def _get_explanation_prompt_template(self) -> str:
        """获取解释提示模板"""
        return """用户请求：{user_message}

需要解释的代码：
```cuda
{cuda_code}
```

```python
{triton_code}
```

上下文信息：
{context}

请详细解释代码的工作原理，包括：
1. 代码整体结构
2. 关键算法实现
3. 性能考虑因素
4. 可能的改进方向

请用中文进行详细说明。""" 