# -*- coding: utf-8 -*-
"""
CUDA到Triton代码生成模块（基于LLM提供商架构）

该模块使用统一的LLM提供商接口，支持多种模型进行CUDA到Triton代码的转换，
并提供多轮对话、错误反馈和自动测试功能。
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
MAX_ITERATIONS = 5
BASE_SLEEP_TIME = 3
# 导入LLM提供商系统
try:
    import dotenv
    from llm.providers import (
        get_provider, PlatformType, ChatRequest, Message
    )
    from llm.providers.impl import GeminiChatTree, OpenRouterChatTree
    from eval_triton_v2_noncheck import TritonKernelEvaluator
    import prompt
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保安装了必要的依赖包并正确配置项目路径")
    sys.exit(1)

# 加载环境变量
dotenv.load_dotenv()


class TritonCodeGenerator:
    """Triton代码生成器"""
    
    def __init__(
        self, 
        platform_type: PlatformType = PlatformType.GOOGLE_OFFICIAL,
        model_name: str = "gemini-2.5-pro",
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
        self.logger = logger or self._create_default_logger()
        self.provider = None
        self.evaluator = TritonKernelEvaluator(logger=self.logger)
        
        # 配置参数
        self.max_retries = MAX_RETRIES
        self.max_iterations = MAX_ITERATIONS
        self.base_sleep_time = BASE_SLEEP_TIME
        
    def _create_default_logger(self) -> logging.Logger:
        """创建默认日志记录器"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                # '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                '[%(levelname)s] %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
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
                    
                    # 将思考和回答都添加到树中
                    chat_tree.add_assistant_message(
                        text=answer_content, 
                        thought=thought_content
                    )
                    
                    # 返回格式化的完整回复
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
        test_dir: str, 
        time_str: str, 
        iteration: Optional[int] = None, 
        timestamp_log_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """测试Triton内核的编译和运行正确性
        
        Args:
            test_dir: 测试目录
            time_str: 时间戳字符串
            iteration: 迭代次数
            timestamp_log_dir: 时间戳日志目录
            
        Returns:
            测试结果字典
        """
        try:
            # 如果提供了迭代信息，添加到日志前缀中
            logfile_prefix = f"round{iteration}_" if iteration is not None else ""
            
            result = await self.evaluator.evaluate_triton_kernel(
                base_dir=test_dir,
                model_name=self.model_name,
                time_str=time_str,
                logfile_prefix=logfile_prefix,
                timestamp_log_dir=timestamp_log_dir,
                return_result=True
            )
            return result
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def setup_logging_dir(self, test_dir: str, time_str: str) -> tuple:
        """设置日志目录
        
        Args:
            test_dir: 测试目录
            time_str: 时间戳字符串
            
        Returns:
            (时间戳日志目录, 对话日志路径)
        """
        log_dir = f"{test_dir}/logs"
        model_name_clean = self.model_name.replace('.', '_').replace('/', '_').replace('-', '_')
        model_log_dir = f"{log_dir}/{model_name_clean}"
        
        # 创建时间戳文件夹
        timestamp_log_dir = f"{model_log_dir}/{time_str}"
        conversation_log_path = f"{timestamp_log_dir}/conversation.log"
        
        os.makedirs(timestamp_log_dir, exist_ok=True)
        
        return timestamp_log_dir, conversation_log_path
    
    def write_conversation_log(
        self, 
        chat_tree, 
        conversation_log_path: str, 
        test_dir: str, 
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
        timestamp_log_dir: str, 
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
            file_path = f"{timestamp_log_dir}/triton_final.py"
            # 同时保存为主要工作文件
            main_file_path = f"{timestamp_log_dir}/triton.py"
            with open(main_file_path, "w") as f:
                f.write(code)
        else:
            file_path = f"{timestamp_log_dir}/triton_round{iteration}.py"
        
        with open(file_path, "w") as f:
            f.write(code)
        
        self.logger.info(f"Code saved to: {file_path}")
        return file_path
    
    def create_code_version_summary(
        self, 
        timestamp_log_dir: str, 
        total_rounds: int, 
        success: bool = True
    ) -> None:
        """创建代码版本摘要文件
        
        Args:
            timestamp_log_dir: 时间戳日志目录
            total_rounds: 总轮次
            success: 是否成功
        """
        summary_path = f"{timestamp_log_dir}/code_versions_summary.md"
        
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(f"# Triton code generation version summary\n\n")
                f.write(f"**Generation time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Model**: {self.model_name}\n")
                f.write(f"**Platform**: {self.platform_type}\n")
                f.write(f"**Total rounds**: {total_rounds}\n")
                f.write(f"**Final result**: {'✅ Success' if success else '❌ Failed'}\n\n")
                
                f.write("## Code version list\n\n")
                
                # 列出所有轮次的代码文件
                for round_num in range(1, total_rounds + 1):
                    round_file = f"triton_round{round_num}.py"
                    f.write(f"- **Round {round_num}**: `{round_file}`\n")
                
                if success:
                    f.write(f"- **Final success version**: `triton_final.py` ✅\n")
                else:
                    f.write(f"- **Last attempt version**: `triton_round{total_rounds}.py` ❌\n")
                
                f.write(f"\n## Related files\n\n")
                f.write(f"- **Conversation log**: `conversation.log`\n")
                f.write(f"- **Evaluation log**: See the corresponding eval log file\n")
                f.write(f"- **Archive version**: `triton.py`\n")
            
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
        
        # 设置目录
        timestamp_log_dir, conversation_log_path = self.setup_logging_dir(test_dir, time_str)
        
        # 读取CUDA代码
        cuda_file_path = f"{test_dir}/cuda_ref.cu"
        try:
            with open(cuda_file_path, "r") as f:
                cuda_code = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"CUDA reference file not found: {cuda_file_path}")
        
        # 创建聊天树
        system_prompt = 'You are a professional GPU computing optimization expert, proficient in CUDA and Triton programming.'
        chat_tree = self.create_chat_tree(system_prompt)
        
        # 第一轮：初始生成请求
        initial_prompt = prompt.complex_initial_prompt.format(cuda_code=cuda_code)
        chat_tree.add_user_message(initial_prompt)
        
        # 记录初始prompt
        self.write_conversation_log(chat_tree, conversation_log_path, test_dir, "Initial prompt generated")
        
        generated_code = ""
        
        # 多轮对话循环
        for iteration in range(self.max_iterations):
            self.logger.info(f"Round {iteration + 1} generation...")
            
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
                chat_tree, conversation_log_path, test_dir, 
                f"Round {iteration + 1} - Model replied"
            )
            
            # 提取代码
            generated_code = self.extract_code_from_response(response)
            
            # 保存当前轮次的代码
            self.save_code_version(generated_code, timestamp_log_dir, iteration + 1)
            
            # 测试代码
            self.logger.info("Testing generated code...")
            test_result = await self.test_triton_kernel(
                test_dir, time_str, iteration + 1, timestamp_log_dir
            )
            
            if test_result.get("success", False):
                self.logger.info("✅ Code test passed!")
                
                # 保存最终成功版本
                self.save_code_version(generated_code, timestamp_log_dir, iteration + 1, is_final=True)
                
                # 打印性能对比结果
                if "performance" in test_result:
                    self.logger.info(f"Performance comparison result:")
                    perf_info = test_result["performance"]
                    if isinstance(perf_info, dict):
                        for key, value in perf_info.items():
                            self.logger.info(f"  {key}: {value}")
                
                # 记录成功结果到日志
                success_info = f"Round {iteration + 1} - Code test passed!\n"
                if "performance" in test_result:
                    success_info += f"Performance result: {test_result['performance']}"
                self.write_conversation_log(chat_tree, conversation_log_path, test_dir, success_info)
                
                # 创建代码版本摘要
                self.create_code_version_summary(timestamp_log_dir, iteration + 1, success=True)
                
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
                self.write_conversation_log(chat_tree, conversation_log_path, test_dir, error_info_log)
        
        # 如果所有迭代都失败了，仍然保存最后一次的代码
        if generated_code:
            last_attempt_path = f"{timestamp_log_dir}/triton_last_attempt.py"
            with open(last_attempt_path, "w") as f:
                f.write(generated_code)
            with open(f"{timestamp_log_dir}/triton.py", "w") as f:
                f.write(generated_code)
            self.logger.info(f"Last attempt code saved to: {last_attempt_path}")
        
        # 记录最终失败日志
        final_log = f"All {self.max_iterations} iterations failed, saved the last generated code"
        self.write_conversation_log(chat_tree, conversation_log_path, test_dir, final_log)
        
        # 创建代码版本摘要
        self.create_code_version_summary(timestamp_log_dir, self.max_iterations, success=False)
        
        return generated_code
    


async def process_single_kernel(generator_config, test_dir, semaphore, logger: Optional[logging.Logger] = None):
    """处理单个内核的生成和评估
    
    Args:
        generator_config: TritonCodeGenerator配置字典
        test_dir: 测试目录路径
        semaphore: 并发限制信号量
        logger: 日志记录器
    
    Returns:
        处理结果字符串
    """
    async with semaphore:  # 限制并发数量
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # 添加微秒避免时间戳冲突
        
        # 为每个任务创建独立的generator实例，避免状态共享
        generator = TritonCodeGenerator(
            platform_type=generator_config["platform_type"],
            model_name=generator_config["model_name"]
        )
        
        try:
            # 初始化generator
            await generator.initialize()
            
            # 设置任务专用的日志记录器，避免日志竞争
            task_logger = logging.getLogger(f"TritonGen-{os.path.basename(test_dir)}-{time_str}")
            task_logger.setLevel(logging.INFO)
            
            # 清除之前的处理器
            for handler in task_logger.handlers[:]:
                task_logger.removeHandler(handler)
            
            # 设置控制台处理器
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                f'[{os.path.basename(test_dir)}] %(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            task_logger.addHandler(console_handler)
            
            # 设置任务专用日志文件
            task_log_file = f"{test_dir}/logs/task_{time_str}.log"
            os.makedirs(os.path.dirname(task_log_file), exist_ok=True)
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
            
            task_logger.info(f"🚀 Start processing {test_dir}")
            await generator.generate_triton_kernel_with_feedback(test_dir, time_str)
            
            # 运行独立的评估
            model_name_clean = generator.model_name.replace('.', '_').replace('/', '_').replace('-', '_')
            timestamp_log_dir = f"{test_dir}/logs/{model_name_clean}/{time_str}"
            
            await generator.evaluator.evaluate_triton_kernel(
                base_dir=test_dir,
                model_name=generator.model_name,
                time_str=time_str,
                timestamp_log_dir=timestamp_log_dir
            )
            
            result_msg = f"✅ Successfully processed {os.path.basename(test_dir)}"
            task_logger.info(result_msg)
            return result_msg
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error processing {os.path.basename(test_dir)}: {e}"
            task_logger.error(error_msg)
            task_logger.error(traceback.format_exc())
            raise Exception(f"{error_msg}\n{traceback.format_exc()}")
        
        finally:
            # 确保generator被正确关闭
            try:
                await generator.close()
            except Exception as close_error:
                if logger:
                    logger.warning(f"Error closing generator for {test_dir}: {close_error}")


async def batch_generate_kernels(levels: List[str] = None, max_concurrent: int = 10, logger: Optional[logging.Logger] = None):
    """批量生成内核代码（并发执行）
    
    Args:
        levels: 要处理的级别列表，默认为['level1']
        max_concurrent: 最大并发数量，默认为10
        logger: 日志记录器
    """
    if levels is None:
        levels = ['level1']
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 准备generator配置（避免共享状态）
    generator_config = {
        "platform_type": PlatformType.GOOGLE_OFFICIAL,
        "model_name": "gemini-2.5-pro"
    }
    
    # 创建并发限制信号量
    semaphore = asyncio.Semaphore(max_concurrent)
    
    try:
        for level in levels:
            output_dir = f"outputs/kernelbench_c/{level}"
            if not os.path.exists(output_dir):
                logger.error(f"Directory does not exist: {output_dir}")
                continue
                
            all_dirs = sorted(os.listdir(output_dir), key=lambda x: int(x.split('_')[0]))
            
            # 创建并发任务列表
            tasks = []
            for dir_name in all_dirs:
                test_dir = f"{output_dir}/{dir_name}"
                # 创建任务协程（注意不要立即执行）
                task_coro = process_single_kernel(generator_config, test_dir, semaphore, logger)
                tasks.append(task_coro)
            
            logger.info(f"🎯 Start concurrent processing {level} level {len(tasks)} kernels (max concurrent: {max_concurrent})")
            
            # 并发执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 统计并处理结果
            success_count = 0
            error_count = 0
            
            logger.info(f"\n📊 {level} level processing result:")
            for i, result in enumerate(results):
                dir_name = all_dirs[i]
                if isinstance(result, Exception):
                    logger.error(f"  ❌ {dir_name}: {str(result).split('\\n')[0]}")
                    error_count += 1
                else:
                    logger.info(f"  {result}")
                    success_count += 1
            
            logger.info(f"\n🏆 {level} level summary: {success_count} passed, {error_count} failed")
            logger.info("-" * 60)
                    
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        import traceback
        logger.error(traceback.format_exc())


# 主函数示例
async def main():
    """主函数示例"""
    # # 示例1: 单个内核生成
    # test_dir = "outputs/kernelbench_c/level1/1_Square_matrix_multiplication_"
    # if os.path.exists(test_dir):
    #     try:
    #         generator = TritonCodeGenerator(
    #             platform_type=PlatformType.GOOGLE_OFFICIAL,
    #             model_name="gemini-2.5-pro"
    #         )
    #         await generator.initialize()
            
    #         code = await generator.generate_triton_kernel_with_feedback(test_dir)
    #         print("✅ 代码生成完成")
    #         print(f"生成的代码长度: {len(code)} 字符")
            
    #     except Exception as e:
    #         print(f"❌ 生成失败: {e}")
    
    # 示例2: 批量生成（取消注释以启用）
    await batch_generate_kernels(['level1'])


if __name__ == "__main__":
    asyncio.run(main())
