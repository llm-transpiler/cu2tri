# -*- coding: utf-8 -*-
"""
CUDA到Triton代码生成模块（基于LLM提供商架构）

该模块使用统一的LLM提供商接口，支持多种模型进行CUDA到Triton代码的转换，
并提供多轮对话、错误反馈和自动测试功能。重写版本，大量复用eval_triton.py的方法。
"""

import os
import sys
import asyncio
import logging
import importlib.util
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# 导入项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

MAX_RETRIES = 5
MAX_ITERATIONS = 3  # 减少到3轮迭代
BASE_SLEEP_TIME = 3
DEFAULT_MODEL = "gemini-2.5-flash"
# 导入LLM提供商系统和eval_triton模块
try:
    import dotenv
    from llm.providers import (
        get_provider, PlatformType, ChatRequest, Message
    )
    from llm.providers.impl import GeminiChatTree, OpenRouterChatTree
    from eval_triton import TritonKernelEvaluator  # 直接使用eval_triton中的评估器
    import prompt
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保安装了必要的依赖包并正确配置项目路径")
    sys.exit(1)

# 加载环境变量
dotenv.load_dotenv()


class TritonCodeGenerator:
    """Triton代码生成器 - 重写版本，大量复用eval_triton.py的方法"""
    
    def __init__(
        self, 
        platform_type: PlatformType = PlatformType.GOOGLE_OFFICIAL,
        model_name: str = DEFAULT_MODEL,
        logger: Optional[logging.Logger] = None
    ):
        """初始化代码生成器
        
        Args:
            platform_type: LLM平台类型
            model_name: 模型名称
            logger: 日志记录器
        """
        self.platform_type = platform_type
        self.model_name = model_name
        self.logger = logger or self._create_logger()
        self.provider = None
        # 直接使用eval_triton.py中的TritonKernelEvaluator
        self.evaluator = TritonKernelEvaluator(logger=self.logger)
        
        # 配置参数
        self.max_retries = MAX_RETRIES
        self.max_iterations = MAX_ITERATIONS
        self.base_sleep_time = BASE_SLEEP_TIME
        
    def _create_logger(self) -> logging.Logger:
        """创建日志记录器 - 复用eval_triton.py的日志创建逻辑"""
        logger = logging.getLogger(__name__)
        
        # 清除现有处理器避免重复
        logger.handlers.clear()
        
        logger.setLevel(logging.DEBUG)
        # 防止传播到根日志记录器避免重复消息
        logger.propagate = False
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-5s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_file_logging(self, base_dir: Path, model_name: str, time_str: str, 
                           logfile_prefix: str = "", timestamp_log_dir: Optional[Path] = None) -> str:
        """设置文件日志 - 复用eval_triton.py的日志设置方法"""
        model_name_clean = model_name.replace('.', '_').replace('/', '_').replace('-', '_')
        
        if timestamp_log_dir:
            log_file = timestamp_log_dir / f"{logfile_prefix}generation.log"
        else:
            log_dir = base_dir / "logs" / model_name_clean
            os.makedirs(log_dir, exist_ok=True)
            log_file = log_dir / f"{logfile_prefix}generation_{time_str}.log"
        
        os.makedirs(log_file.parent, exist_ok=True)
        
        # 检查文件处理器是否已存在
        file_handler_exists = False
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file.resolve()):
                file_handler_exists = True
                break
        
        # 只有在不存在时才添加文件处理器
        if not file_handler_exists:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-5s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        return str(log_file)
    
    async def initialize(self) -> None:
        """初始化LLM提供商"""
        try:
            self.provider = get_provider(self.platform_type)
            self.logger.info(f"✅ LLM provider initialized: {self.platform_type}")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM provider: {e}")
            raise
    
    async def close(self) -> None:
        """关闭LLM提供商连接"""
        if self.provider:
            try:
                await self.provider.close()
                self.logger.info("✅ LLM provider connection closed")
            except Exception as e:
                self.logger.warning(f"⚠️ Error closing LLM provider connection: {e}")
    
    def create_chat_tree(self, system_prompt: Optional[str] = None):
        """创建聊天树对象
        
        Args:
            system_prompt: 系统提示词
            
        Returns:
            相应的ChatTree对象
        """
        if self.platform_type == PlatformType.GOOGLE_OFFICIAL:
            return GeminiChatTree(system_prompt=system_prompt)
        elif self.platform_type == PlatformType.OPENROUTER:
            return OpenRouterChatTree(system_prompt=system_prompt)
        else:
            # 对于其他平台，使用默认的OpenRouterChatTree
            return OpenRouterChatTree(system_prompt=system_prompt)
    
    async def generate_with_retries(self, chat_tree, request: ChatRequest) -> str:
        """带重试的生成请求
        
        Args:
            chat_tree: 聊天树对象
            request: 聊天请求
            
        Returns:
            生成的响应内容
            
        Raises:
            Exception: 当所有重试都失败时
        """
        for retry in range(self.max_retries):
            try:
                response = await self.provider.chat(request)
                
                # 根据平台类型处理思考内容
                if hasattr(response, 'thoughts') and response.thoughts:
                    thought_content = response.thoughts
                    answer_content = response.content
                    #  # 将思考和回答都添加到树中
                    # chat_tree.add_assistant_message(
                    #     text=answer_content, 
                    #     thought=thought_content
                    # )
                    # 只将答案部分添加到对话历史中，thinking部分不参与下一轮对话
                    chat_tree.add_assistant_message(answer_content)
                    
                    # 返回格式化的完整回复（用于日志记录）
                    return f"<thoughts>{thought_content}</thoughts><answer>{answer_content}</answer>"
                else:
                    # 没有思考内容，只添加回答
                    chat_tree.add_assistant_message(response.content)
                    return response.content
                    
            except Exception as e:
                self.logger.warning(f"Generation request failed (retry {retry + 1}/{self.max_retries}): {e}")
                if retry < self.max_retries - 1:
                    await asyncio.sleep(self.base_sleep_time * (2 ** retry))
                else:
                    raise Exception(f"All retries failed: {e}")
    
    def extract_code_from_response(self, response: str) -> str:
        """从响应中提取Python代码
        
        Args:
            response: LLM响应内容
            
        Returns:
            提取的代码字符串
        """
        # 首先尝试提取<answer></answer>中的内容
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1)
            
            # 在answer内容中尝试提取```python代码块
            code_match = re.search(r'```python\n(.*?)\n```', answer_content, re.DOTALL)
            if code_match:
                return code_match.group(1)
            
            # 在answer内容中尝试提取```代码块
            code_match = re.search(r'```\n(.*?)\n```', answer_content, re.DOTALL)
            if code_match:
                return code_match.group(1)
            
            # 如果answer中没有代码块，返回answer的全部内容
            return answer_content
        
        # 如果没有找到answer标签，使用原有逻辑
        # 尝试提取```python代码块
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # 尝试提取```代码块
        code_match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # 如果没有找到代码块，返回整个响应
        return response
    
    async def test_triton_kernel(
        self, 
        base_dir: Path, 
        time_str: str, 
        iteration: Optional[int] = None, 
        timestamp_log_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """测试Triton内核 - 直接使用eval_triton.py中的evaluate_triton_kernel方法
        
        Args:
            base_dir: 测试目录
            time_str: 时间戳字符串
            iteration: 迭代次数
            timestamp_log_dir: 时间戳日志目录
            
        Returns:
            测试结果字典
        """
        try:
            # 如果提供了迭代信息，添加到日志前缀中
            logfile_prefix = f"round{iteration}_" if iteration is not None else ""
            
            # 直接调用eval_triton.py中的方法
            result = await self.evaluator.evaluate_triton_kernel(
                base_dir=base_dir,
                model_name=self.model_name,
                time_str=time_str,
                logfile_prefix=logfile_prefix,
                timestamp_log_dir=timestamp_log_dir,
                return_result=True,
                timeout=300,
                capture_output=True
            )
            return result
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def setup_logging_dir(self, test_dir: Path, time_str: str) -> tuple:
        """设置日志目录 - 使用Path对象，与eval_triton.py保持一致
        
        Args:
            test_dir: 测试目录
            time_str: 时间戳字符串
            
        Returns:
            (时间戳日志目录Path对象, 对话日志路径字符串)
        """
        log_dir = test_dir / "logs"
        model_name_clean = self.model_name.replace('.', '_').replace('/', '_').replace('-', '_')
        model_log_dir = log_dir / model_name_clean
        
        # 创建时间戳文件夹
        timestamp_log_dir = model_log_dir / time_str
        conversation_log_path = timestamp_log_dir / "conversation.log"
        
        os.makedirs(timestamp_log_dir, exist_ok=True)
        
        return timestamp_log_dir, str(conversation_log_path)
    
    def write_conversation_log(
        self, 
        chat_tree, 
        conversation_log_path: str, 
        test_dir: Path, 
        iteration_info: str = ""
    ) -> None:
        """将对话内容写入日志文件
        
        Args:
            chat_tree: 聊天树对象
            conversation_log_path: 对话日志文件路径
            test_dir: 测试目录
            iteration_info: 迭代信息
        """
        try:
            with open(conversation_log_path, "w", encoding="utf-8") as f:
                f.write(f"=== CUDA to Triton code conversion conversation log ===\n")
                f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Test directory: {test_dir}\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Platform: {self.platform_type}\n")
                f.write(f"Maximum iterations: {self.max_iterations}\n")
                if iteration_info:
                    f.write(f"{iteration_info}\n")
                f.write(f"="*60 + "\n\n")
                
                # 写入对话历史
                f.write(chat_tree.get_history().simple_print())
                f.write(f"\n{'='*60}\n\n")
        except Exception as e:
            self.logger.warning(f"Failed to write conversation log: {e}")
    
    def save_code_version(
        self, 
        code: str, 
        timestamp_log_dir: Path, 
        iteration: int,
        is_final: bool = False
    ) -> str:
        """保存代码版本
        
        Args:
            code: 代码内容
            timestamp_log_dir: 时间戳日志目录
            iteration: 迭代次数
            is_final: 是否为最终版本
            
        Returns:
            保存的文件路径
        """
        if is_final:
            file_path = timestamp_log_dir / "triton_final.py"
            # 同时保存为主要工作文件，与eval_triton.py的文件命名保持一致
            main_file_path = timestamp_log_dir / "triton_ref.py"
            with open(main_file_path, "w") as f:
                f.write(code)
        else:
            file_path = timestamp_log_dir / f"triton_round{iteration}.py"
        
        with open(file_path, "w") as f:
            f.write(code)
        
        self.logger.info(f"Code saved to: {file_path}")
        return str(file_path)
    
    def create_code_version_summary(
        self, 
        timestamp_log_dir: Path, 
        total_rounds: int, 
        success: bool = True
    ) -> None:
        """创建代码版本摘要文件
        
        Args:
            timestamp_log_dir: 时间戳日志目录
            total_rounds: 总轮次
            success: 是否成功
        """
        summary_path = timestamp_log_dir / "code_versions_summary.md"
        
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(f"# Triton code generation version summary\n\n")
                f.write(f"**Generation time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Model**: {self.model_name}\n")
                f.write(f"**Platform**: {self.platform_type}\n")
                f.write(f"**Total rounds**: {total_rounds}\n")
                f.write(f"**Maximum iterations**: {self.max_iterations}\n")
                f.write(f"**Final result**: {'✅ Success' if success else '❌ Failed'}\n\n")
                
                f.write("## Code version list\n\n")
                
                # 列出所有轮次的代码文件
                for round_num in range(1, total_rounds + 1):
                    round_file = f"triton_round{round_num}.py"
                    f.write(f"- **Round {round_num}**: `{round_file}`\n")
                
                if success:
                    f.write(f"- **Final success version**: `triton_final.py` ✅\n")
                    f.write(f"- **Working version**: `triton_ref.py` ✅\n")
                else:
                    f.write(f"- **Last attempt version**: `triton_round{total_rounds}.py` ❌\n")
                
                f.write(f"\n## Related files\n\n")
                f.write(f"- **Conversation log**: `conversation.log`\n")
                f.write(f"- **Evaluation log**: See the corresponding eval log file\n")
                f.write(f"- **Generation log**: `generation.log`\n")
            
            self.logger.info(f"Code version summary created: {summary_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create code version summary: {e}")
    
    async def generate_triton_kernel_with_feedback(
        self, 
        test_dir: str, 
        time_str: Optional[str] = None
    ) -> str:
        """通过多轮对话和反馈生成Triton内核代码
        
        Args:
            test_dir: 测试目录
            time_str: 时间戳字符串
            
        Returns:
            生成的Triton代码
        """
        # 确保初始化
        if not self.provider:
            await self.initialize()
        
        # 设置时间戳
        if time_str is None:
            time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 转换为Path对象，与eval_triton.py保持一致
        test_dir_path = Path(test_dir)
        
        # 设置目录和日志
        timestamp_log_dir, conversation_log_path = self.setup_logging_dir(test_dir_path, time_str)
        log_file = self._setup_file_logging(test_dir_path, self.model_name, time_str, timestamp_log_dir=timestamp_log_dir)
        
        # 开始信息日志 - 复用eval_triton.py的日志格式
        self.logger.info("")
        self.logger.info("╔" + "═" * 58 + "╗")
        self.logger.info("║" + f"{'Triton Code Generation Started':^58}" + "║")
        self.logger.info("║" + f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^58}" + "║")
        self.logger.info("╚" + "═" * 58 + "╝")
        
        # 读取CUDA代码
        cuda_file_path = test_dir_path / "cuda_ref.cu"
        try:
            with open(cuda_file_path, "r") as f:
                cuda_code = f.read()
        except FileNotFoundError:
            error_msg = f"CUDA reference file not found: {cuda_file_path}"
            self.logger.error(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)
        
        # 创建聊天树
        system_prompt = 'You are a professional GPU computing optimization expert, proficient in CUDA and Triton programming.'
        chat_tree = self.create_chat_tree(system_prompt)
        
        # 第一轮：初始生成请求
        initial_prompt = prompt.complex_initial_prompt.format(cuda_code=cuda_code)
        chat_tree.add_user_message(initial_prompt)
        
        # 记录初始prompt
        self.write_conversation_log(chat_tree, conversation_log_path, test_dir_path, "Initial prompt generated")
        
        self.logger.info("📁 File Paths:")
        self.logger.info(f"    CUDA:     {cuda_file_path}")
        self.logger.info(f"    Log Dir:  {timestamp_log_dir}")
        self.logger.info("")
        
        generated_code = ""
        
        # 多轮对话循环 (现在最多3轮)
        for iteration in range(self.max_iterations):
            self.logger.info(f"🔄 Round {iteration + 1}/{self.max_iterations} generation...")
            
            # 创建请求
            request = ChatRequest(
                messages=chat_tree.get_history().messages,
                model=self.model_name,
                include_thinking=True,
                thinking_budget=-1,
                temperature=0.3,
                max_tokens=8192 * 2
            )
            
            # 生成代码
            try:
                response = await self.generate_with_retries(chat_tree, request)
            except Exception as e:
                self.logger.error(f"Round {iteration + 1} generation failed: {e}")
                continue
            
            # 更新对话日志
            self.write_conversation_log(
                chat_tree, conversation_log_path, test_dir_path, 
                f"Round {iteration + 1} - Model replied"
            )
            
            # 提取代码
            generated_code = self.extract_code_from_response(response)
            
            # 保存当前轮次的代码
            self.save_code_version(generated_code, timestamp_log_dir, iteration + 1)
            
            # 测试代码 - 使用eval_triton.py的方法
            self.logger.info("🧪 Testing generated code...")
            test_result = await self.test_triton_kernel(
                test_dir_path, time_str, iteration + 1, timestamp_log_dir
            )
            
            if test_result.get("success", False):
                self.logger.info("✅ Code test passed!")
                
                # 保存最终成功版本
                self.save_code_version(generated_code, timestamp_log_dir, iteration + 1, is_final=True)
                
                # 打印性能对比结果 - 复用eval_triton.py的格式
                if "performance" in test_result:
                    self.logger.info("📊 Performance comparison result:")
                    perf_info = test_result["performance"]
                    if isinstance(perf_info, dict):
                        triton_time = perf_info.get('triton_time', 0)
                        cuda_time = perf_info.get('cuda_time', 0)
                        torch_time = perf_info.get('torch_time', 0)
                        triton_cuda_speedup = perf_info.get('triton_cuda_speedup', 0)
                        triton_torch_speedup = perf_info.get('triton_torch_speedup', 0)
                        
                        self.logger.info("─" * 50)
                        self.logger.info(f"    Triton:   {triton_time:8.3f} ms")
                        self.logger.info(f"    CUDA:     {cuda_time:8.3f} ms")
                        self.logger.info(f"    PyTorch:  {torch_time:8.3f} ms")
                        self.logger.info("─" * 30)
                        self.logger.info(f"    Speedup (vs CUDA):    {triton_cuda_speedup:6.2f}x")
                        self.logger.info(f"    Speedup (vs PyTorch): {triton_torch_speedup:6.2f}x")
                
                # 记录成功结果到日志
                success_info = f"Round {iteration + 1} - Code test passed!\n"
                if "performance" in test_result:
                    success_info += f"Performance result: {test_result['performance']}"
                self.write_conversation_log(chat_tree, conversation_log_path, test_dir_path, success_info)
                
                # 创建代码版本摘要
                self.create_code_version_summary(timestamp_log_dir, iteration + 1, success=True)
                
                # 成功完成日志 - 复用eval_triton.py的格式
                self.logger.info("")
                self.logger.info("╔" + "═" * 58 + "╗")
                self.logger.info("║" + f"{'🎉 Generation Completed Successfully!':^57}" + "║")
                self.logger.info("╚" + "═" * 58 + "╝")
                self.logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info("")
                
                return generated_code
            else:
                self.logger.info(f"❌ Code test failed: {test_result.get('error', 'Unknown error')}")
                
                if iteration == self.max_iterations - 1:
                    self.logger.info(f"Reached maximum iteration ({self.max_iterations}), stopping attempt")
                    break
                
                # 准备反馈信息
                error_info = test_result.get('error', 'Unknown error')
                traceback_info = test_result.get('traceback', '')
                
                feedback_prompt = prompt.feedback_prompt.format(
                    error_info=error_info, 
                    traceback_info=traceback_info
                )
                
                chat_tree.add_user_message(feedback_prompt)
                
                # 记录反馈信息到日志
                error_info_log = f"Round {iteration + 1} - Code test failed, provided error feedback\nError: {error_info}"
                self.write_conversation_log(chat_tree, conversation_log_path, test_dir_path, error_info_log)
        
        # 如果所有迭代都失败了，仍然保存最后一次的代码
        if generated_code:
            last_attempt_path = timestamp_log_dir / "triton_last_attempt.py"
            with open(last_attempt_path, "w") as f:
                f.write(generated_code)
            # 也保存为主要工作文件
            main_file_path = timestamp_log_dir / "triton_ref.py"
            with open(main_file_path, "w") as f:
                f.write(generated_code)
            self.logger.info(f"Last attempt code saved to: {last_attempt_path}")
        
        # 记录最终失败日志
        final_log = f"All {self.max_iterations} iterations failed, saved the last generated code"
        self.write_conversation_log(chat_tree, conversation_log_path, test_dir_path, final_log)
        
        # 创建代码版本摘要
        self.create_code_version_summary(timestamp_log_dir, self.max_iterations, success=False)
        
        # 失败完成日志 - 复用eval_triton.py的格式
        self.logger.error("")
        self.logger.error("╔" + "═" * 58 + "╗")
        self.logger.error("║" + f"{'❌ Generation Failed':^57}" + "║")
        self.logger.error("╚" + "═" * 58 + "╝")
        self.logger.error(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.error("")
        
        return generated_code


async def process_single_kernel(generator_config, test_dir, semaphore, global_time_str: str, logger: Optional[logging.Logger] = None):
    """处理单个内核的生成和评估 - 优化版本，使用eval_triton.py的方法
    
    Args:
        generator_config: TritonCodeGenerator配置字典
        test_dir: 测试目录路径
        semaphore: 并发限制信号量
        global_time_str: 全局时间戳，用于标识整个批次
        logger: 日志记录器
    
    Returns:
        处理结果字符串
    """
    # 使用全局时间戳 + 任务特定标识符
    task_time_str = f"{global_time_str}_{Path(test_dir).name}"
    
    # 为每个任务创建独立的generator实例，避免状态共享
    generator = TritonCodeGenerator(
        platform_type=generator_config["platform_type"],
        model_name=generator_config["model_name"]
    )
    
    try:
        # 初始化generator - 放在semaphore外面，避免初始化阻塞并发
        await generator.initialize()
        
        # 设置任务专用的日志记录器，避免日志竞争
        task_logger = logging.getLogger(f"TritonGen-{Path(test_dir).name}-{task_time_str}")
        task_logger.setLevel(logging.INFO)
        
        # 清除之前的处理器
        for handler in task_logger.handlers[:]:
            task_logger.removeHandler(handler)
        
        # 设置控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            f'[{Path(test_dir).name}][{global_time_str}] |%(levelname)s|\t%(message)s'
        )
        console_handler.setFormatter(console_formatter)
        task_logger.addHandler(console_handler)
        
        # 设置任务专用日志文件
        task_log_file = Path(test_dir) / "logs" / f"task_{task_time_str}.log"
        os.makedirs(task_log_file.parent, exist_ok=True)
        file_handler = logging.FileHandler(task_log_file, mode='w')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        task_logger.addHandler(file_handler)
        
        # 防止日志向上传播
        task_logger.propagate = False
        
        # 更新generator的logger
        generator.logger = task_logger
        generator.evaluator.logger = task_logger
        
        async with semaphore:  # 限制并发数量，主要限制实际的LLM调用和GPU资源
            start_time = datetime.now()
            task_logger.info(f"🚀 Start processing {test_dir} (Batch: {global_time_str}) at {start_time.strftime('%H:%M:%S')}")
            await generator.generate_triton_kernel_with_feedback(test_dir, global_time_str)
            
            # 运行独立的评估 - 使用eval_triton.py的方法
            model_name_clean = generator.model_name.replace('.', '_').replace('/', '_').replace('-', '_')
            timestamp_log_dir = Path(test_dir) / "logs" / model_name_clean / global_time_str
            
            await generator.evaluator.evaluate_triton_kernel(
                base_dir=Path(test_dir),
                model_name=generator.model_name,
                time_str=global_time_str,
                timestamp_log_dir=timestamp_log_dir,
                return_result=False,
                timeout=300,
                capture_output=True
            )
            
            end_time = datetime.now()
            duration = end_time - start_time
            result_msg = f"✅ Successfully processed {Path(test_dir).name} in {duration.total_seconds():.2f}s"
            task_logger.info(result_msg)
            return result_msg
            
    except Exception as e:
        import traceback
        error_msg = f"❌ Error processing {Path(test_dir).name}: {e}"
        if hasattr(generator, 'logger'):
            generator.logger.error(error_msg)
            generator.logger.error(traceback.format_exc())
        raise Exception(f"{error_msg}\n{traceback.format_exc()}")
    
    finally:
        # 确保generator被正确关闭
        try:
            await generator.close()
        except Exception as close_error:
            if logger:
                logger.warning(f"Error closing generator for {test_dir}: {close_error}")


async def batch_generate_kernels(test_dirs: List[str] = None, max_concurrent: int = 10, logger: Optional[logging.Logger] = None):
    """批量生成内核代码（并发执行）- 优化版本
    
    Args:
        test_dirs: 要处理的目录列表，默认为['01_single_op']
        max_concurrent: 最大并发数量，默认为10
        logger: 日志记录器
    """
    if test_dirs is None:
        test_dirs = ['01_single_op']  # 现在固定使用此目录
    
    if logger is None:
        logger = logging.getLogger(__name__)
        # 确保logger有处理器
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | %(levelname)-5s | %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    # 生成全局时间戳，用于标识整个批次的测试
    global_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"📅 Global batch timestamp: {global_time_str}")  # 使用print确保可见
    logger.info(f"📅 Global batch timestamp: {global_time_str}")
    print(f"🔧 Concurrency settings: max_concurrent={max_concurrent}")
    logger.info(f"🔧 Concurrency settings: max_concurrent={max_concurrent}")
    
    # 创建共享的LLM提供商实例，避免重复初始化
    print("🔧 Initializing shared LLM provider...")
    logger.info("🔧 Initializing shared LLM provider...")
    shared_provider = get_provider(PlatformType.GOOGLE_OFFICIAL)
    print("✅ Shared LLM provider initialized")
    logger.info("✅ Shared LLM provider initialized")
    
    # 准备generator配置（避免共享状态）
    generator_config = {
        "platform_type": PlatformType.GOOGLE_OFFICIAL,
        "model_name": DEFAULT_MODEL,
        "shared_provider": shared_provider  # 添加共享提供商
    }
    
    # 创建并发限制信号量
    semaphore = asyncio.Semaphore(max_concurrent)
    print(f"🚦 Created semaphore with {max_concurrent} slots")
    logger.info(f"🚦 Created semaphore with {max_concurrent} slots")
    
    try:
        for sub_dir in test_dirs:
            # 修改为新的输出目录结构
            output_dir = f"outputs/cu2tri/kernelbench_c/{sub_dir}"
            if not os.path.exists(output_dir):
                print(f"❌ Directory does not exist: {output_dir}")
                logger.error(f"Directory does not exist: {output_dir}")
                continue
                
            all_dirs = sorted(os.listdir(output_dir), key=lambda x: int(x.split('_')[0]))
            
            # 限制任务数量用于测试
            test_dirs_list = all_dirs[:3]  # 只测试前3个任务
            print(f"🧪 Testing with first {len(test_dirs_list)} directories: {test_dirs_list}")
            logger.info(f"🧪 Testing with first {len(test_dirs_list)} directories: {test_dirs_list}")
            
            # 创建并发任务列表
            tasks = []
            for dir_name in test_dirs_list:
                test_dir = f"{output_dir}/{dir_name}"
                print(f"📝 Creating task for: {dir_name}")
                logger.info(f"📝 Creating task for: {dir_name}")
                # 创建任务协程（注意传递全局时间戳）
                task_coro = process_single_kernel_with_shared_provider(generator_config, test_dir, semaphore, global_time_str, logger)
                tasks.append(task_coro)
                print(f"✅ Task created for: {dir_name}")
                logger.info(f"✅ Task created for: {dir_name}")
            
            print(f"🎯 Start concurrent processing {sub_dir} {len(tasks)} kernels (batch: {global_time_str}, max concurrent: {max_concurrent}, max iterations: {MAX_ITERATIONS})")
            logger.info(f"🎯 Start concurrent processing {sub_dir} {len(tasks)} kernels (batch: {global_time_str}, max concurrent: {max_concurrent}, max iterations: {MAX_ITERATIONS})")
            print(f"📋 Task list created with {len(tasks)} tasks")
            logger.info(f"📋 Task list created with {len(tasks)} tasks")
            
            # 添加开始时间记录
            start_time = datetime.now()
            print(f"⏰ Batch start time: {start_time.strftime('%H:%M:%S')}")
            logger.info(f"⏰ Batch start time: {start_time.strftime('%H:%M:%S')}")
            print(f"🚀 About to call asyncio.gather() with {len(tasks)} tasks...")
            logger.info(f"🚀 About to call asyncio.gather() with {len(tasks)} tasks...")
            
            # 并发执行所有任务
            print("🎬 Starting asyncio.gather()...")
            
            # 先测试任务是否能同时开始
            async def debug_task_start(task_coro, task_name):
                print(f"🚀 DEBUG: About to start task {task_name}")
                result = await task_coro
                print(f"✅ DEBUG: Task {task_name} completed")
                return result
            
            # 包装任务以便调试
            debug_tasks = []
            for i, task_coro in enumerate(tasks):
                task_name = test_dirs_list[i]
                debug_task = debug_task_start(task_coro, task_name)
                debug_tasks.append(debug_task)
            
            print(f"🎯 About to execute {len(debug_tasks)} debug-wrapped tasks...")
            results = await asyncio.gather(*debug_tasks, return_exceptions=True)
            
            # 添加结束时间记录
            end_time = datetime.now()
            duration = end_time - start_time
            print(f"⏰ Batch end time: {end_time.strftime('%H:%M:%S')}")
            logger.info(f"⏰ Batch end time: {end_time.strftime('%H:%M:%S')}")
            print(f"⌛ Total batch duration: {duration.total_seconds():.2f} seconds")
            logger.info(f"⌛ Total batch duration: {duration.total_seconds():.2f} seconds")
            
            # 统计并处理结果
            success_count = 0
            error_count = 0
            
            print(f"\n📊 {sub_dir} processing result:")
            logger.info(f"\n📊 {sub_dir} processing result:")
            for i, result in enumerate(results):
                dir_name = test_dirs_list[i]
                if isinstance(result, Exception):
                    print(f"  ❌ {dir_name}: {str(result).split('\\n')[0]}")
                    logger.error(f"  ❌ {dir_name}: {str(result).split('\\n')[0]}")
                    error_count += 1
                else:
                    print(f"  {result}")
                    logger.info(f"  {result}")
                    success_count += 1
            
            print(f"\n🏆 {sub_dir} summary: {success_count} passed, {error_count} failed")
            logger.info(f"\n🏆 {sub_dir} summary: {success_count} passed, {error_count} failed")
            print("-" * 60)
            logger.info("-" * 60)
                    
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        logger.error(f"Batch processing error: {e}")
        import traceback
        print(traceback.format_exc())
        logger.error(traceback.format_exc())
    finally:
        # 关闭共享提供商
        try:
            await shared_provider.close()
            print("🔒 Shared LLM provider closed")
            logger.info("🔒 Shared LLM provider closed")
        except Exception as e:
            print(f"⚠️ Error closing shared provider: {e}")
            logger.warning(f"Error closing shared provider: {e}")


async def process_single_kernel_with_shared_provider(generator_config, test_dir, semaphore, global_time_str: str, logger: Optional[logging.Logger] = None):
    """处理单个内核的生成和评估 - 使用共享提供商版本
    
    Args:
        generator_config: TritonCodeGenerator配置字典（包含shared_provider）
        test_dir: 测试目录路径
        semaphore: 并发限制信号量
        global_time_str: 全局时间戳，用于标识整个批次
        logger: 日志记录器
    
    Returns:
        处理结果字符串
    """
    # 使用全局时间戳 + 任务特定标识符
    task_time_str = f"{global_time_str}_{Path(test_dir).name}"
    
    print(f"🔄 Task starting for {Path(test_dir).name} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    
    # 为每个任务创建独立的generator实例，但使用共享的提供商
    generator = TritonCodeGenerator(
        platform_type=generator_config["platform_type"],
        model_name=generator_config["model_name"]
    )
    
    # 直接设置共享的提供商，跳过初始化
    generator.provider = generator_config["shared_provider"]
    print(f"✅ Shared provider assigned to {Path(test_dir).name}")
    
    try:
        # 设置任务专用的日志记录器，避免日志竞争
        task_logger = logging.getLogger(f"TritonGen-{Path(test_dir).name}-{task_time_str}")
        task_logger.setLevel(logging.INFO)
        
        # 清除之前的处理器
        for handler in task_logger.handlers[:]:
            task_logger.removeHandler(handler)
        
        # 设置控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            f'[{Path(test_dir).name}][{global_time_str}] |%(levelname)s|\t%(message)s'
        )
        console_handler.setFormatter(console_formatter)
        task_logger.addHandler(console_handler)
        
        # 设置任务专用日志文件
        task_log_file = Path(test_dir) / "logs" / f"task_{task_time_str}.log"
        os.makedirs(task_log_file.parent, exist_ok=True)
        file_handler = logging.FileHandler(task_log_file, mode='w')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        task_logger.addHandler(file_handler)
        
        # 防止日志向上传播
        task_logger.propagate = False
        
        # 更新generator的logger
        generator.logger = task_logger
        generator.evaluator.logger = task_logger
        
        print(f"🎯 About to acquire semaphore for {Path(test_dir).name}")
        async with semaphore:  # 限制并发数量，主要限制实际的LLM调用和GPU资源
            start_time = datetime.now()
            print(f"🔒 Semaphore acquired for {Path(test_dir).name} at {start_time.strftime('%H:%M:%S.%f')[:-3]}")
            task_logger.info(f"🚀 Start processing {test_dir} (Batch: {global_time_str}) at {start_time.strftime('%H:%M:%S')}")
            await generator.generate_triton_kernel_with_feedback(test_dir, global_time_str)
            
            # 运行独立的评估 - 使用eval_triton.py的方法
            model_name_clean = generator.model_name.replace('.', '_').replace('/', '_').replace('-', '_')
            timestamp_log_dir = Path(test_dir) / "logs" / model_name_clean / global_time_str
            
            await generator.evaluator.evaluate_triton_kernel(
                base_dir=Path(test_dir),
                model_name=generator.model_name,
                time_str=global_time_str,
                timestamp_log_dir=timestamp_log_dir,
                return_result=False,
                timeout=300,
                capture_output=True
            )
            
            end_time = datetime.now()
            duration = end_time - start_time
            result_msg = f"✅ Successfully processed {Path(test_dir).name} in {duration.total_seconds():.2f}s"
            print(f"🎉 Task completed for {Path(test_dir).name} at {end_time.strftime('%H:%M:%S.%f')[:-3]}")
            task_logger.info(result_msg)
            return result_msg
            
    except Exception as e:
        import traceback
        error_msg = f"❌ Error processing {Path(test_dir).name}: {e}"
        print(f"💥 Task failed for {Path(test_dir).name}: {e}")
        if hasattr(generator, 'logger'):
            generator.logger.error(error_msg)
            generator.logger.error(traceback.format_exc())
        raise Exception(f"{error_msg}\n{traceback.format_exc()}")
    
    finally:
        # 不需要关闭shared provider，由主函数负责
        print(f"🧹 Task cleanup for {Path(test_dir).name}")
        pass


# 主函数示例
async def main():
    # 首先测试基本的并发功能
    print("=" * 60)
    print("🧪 Phase 1: Testing basic concurrency")
    print("=" * 60)
    concurrent_works = await test_concurrent_execution()
    
    if not concurrent_works:
        print("❌ Basic concurrency test failed! There might be an issue with the asyncio setup.")
        return
    
    # 测试简单的Triton任务并发
    simple_concurrent_works = await test_triton_task_concurrency()
    
    if not simple_concurrent_works:
        print("❌ Simple Triton task concurrency failed!")
        return
    
    print("\n" + "=" * 60)
    print("🚀 Phase 3: Testing real Triton generation concurrency")
    print("=" * 60)
    
    # 如果基本并发测试通过，再进行实际的批量生成
    await batch_generate_kernels(['01_single_op'])


# 简单的并发测试函数
async def test_concurrent_execution():
    """测试并发执行是否正常工作"""
    
    async def simple_task(task_id: int, delay: float):
        print(f"⏰ Task {task_id} starting at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        await asyncio.sleep(delay)
        print(f"✅ Task {task_id} finished at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        return f"Task {task_id} completed"
    
    print("🧪 Testing concurrent execution...")
    start_time = datetime.now()
    
    # 创建3个任务，每个延迟2秒
    tasks = [
        simple_task(1, 2.0),
        simple_task(2, 2.0), 
        simple_task(3, 2.0)
    ]
    
    print(f"🚀 Starting {len(tasks)} tasks concurrently at {start_time.strftime('%H:%M:%S')}")
    results = await asyncio.gather(*tasks)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"⏰ All tasks completed at {end_time.strftime('%H:%M:%S')}")
    print(f"⌛ Total duration: {duration.total_seconds():.2f} seconds")
    print(f"📊 Results: {results}")
    
    if duration.total_seconds() < 4:  # 如果并发工作，应该约2秒完成，而不是6秒
        print("✅ Concurrent execution is working!")
        return True
    else:
        print("❌ Tasks appear to be running serially!")
        return False


async def simple_triton_task(task_name: str, delay: float = 1.0):
    """简单的Triton任务模拟，用于测试并发"""
    print(f"🔄 Simple task {task_name} starting at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    
    # 模拟一些async工作
    await asyncio.sleep(delay)
    
    print(f"✅ Simple task {task_name} finished at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    return f"Simple task {task_name} completed"


async def test_triton_task_concurrency():
    """测试Triton任务的并发执行"""
    print("\n" + "=" * 60)
    print("🧪 Phase 2.5: Testing simple Triton task concurrency")
    print("=" * 60)
    
    # 创建3个简单任务
    tasks = [
        simple_triton_task("Task1", 2.0),
        simple_triton_task("Task2", 2.0),
        simple_triton_task("Task3", 2.0)
    ]
    
    start_time = datetime.now()
    print(f"🚀 Starting {len(tasks)} simple Triton tasks at {start_time.strftime('%H:%M:%S')}")
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"⏰ All simple tasks completed at {end_time.strftime('%H:%M:%S')}")
    print(f"⌛ Total duration: {duration.total_seconds():.2f} seconds")
    print(f"📊 Results: {results}")
    
    if duration.total_seconds() < 4:
        print("✅ Simple Triton task concurrency is working!")
        return True
    else:
        print("❌ Simple Triton tasks appear to be running serially!")
        return False


if __name__ == "__main__":
    asyncio.run(main())
