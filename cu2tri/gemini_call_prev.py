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
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# 导入项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入LLM提供商系统
try:
    import dotenv
    from llm.providers import (
        get_provider, PlatformType, ChatRequest, Message
    )
    from llm.providers.impl import GeminiChatTree, OpenRouterChatTree
    from eval_triton import TritonKernelEvaluator
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
        self.max_retries = 5
        self.max_iterations = 5
        self.base_sleep_time = 3
        
    def _create_default_logger(self) -> logging.Logger:
        """创建默认日志记录器"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    async def initialize(self) -> None:
        """初始化LLM提供商"""
        try:
            self.provider = get_provider(self.platform_type)
            self.logger.info(f"✅ 已初始化LLM提供商: {self.platform_type}")
        except Exception as e:
            self.logger.error(f"❌ 初始化LLM提供商失败: {e}")
            raise
    
    async def close(self) -> None:
        """关闭LLM提供商连接"""
        if self.provider:
            try:
                await self.provider.close()
                self.logger.info("✅ 已关闭LLM提供商连接")
            except Exception as e:
                self.logger.warning(f"⚠️ 关闭LLM提供商连接时出错: {e}")
    
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
                self.logger.warning(f"生成请求失败 (重试 {retry + 1}/{self.max_retries}): {e}")
                if retry < self.max_retries - 1:
                    await asyncio.sleep(self.base_sleep_time * (2 ** retry))
                else:
                    raise Exception(f"所有重试都失败了: {e}")
    
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
    
    def setup_logging_directories(self, test_dir: str, time_str: str) -> tuple:
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
                f.write(f"=== CUDA到Triton代码转换对话日志 ===\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"测试目录: {test_dir}\n")
                f.write(f"模型: {self.model_name}\n")
                f.write(f"平台: {self.platform_type}\n")
                f.write(f"最大迭代次数: {self.max_iterations}\n")
                if iteration_info:
                    f.write(f"{iteration_info}\n")
                f.write(f"="*60 + "\n\n")
                
                # 写入对话历史
                f.write(chat_tree.get_history().simple_print())
                f.write(f"\n{'='*60}\n\n")
        except Exception as e:
            self.logger.warning(f"写入对话日志失败: {e}")
    
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
        
        self.logger.info(f"代码已保存到: {file_path}")
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
                f.write(f"# Triton代码生成版本摘要\n\n")
                f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**模型**: {self.model_name}\n")
                f.write(f"**平台**: {self.platform_type}\n")
                f.write(f"**总轮次**: {total_rounds}\n")
                f.write(f"**最终结果**: {'✅ 成功' if success else '❌ 失败'}\n\n")
                
                f.write("## 代码版本列表\n\n")
                
                # 列出所有轮次的代码文件
                for round_num in range(1, total_rounds + 1):
                    round_file = f"triton_round{round_num}.py"
                    f.write(f"- **第{round_num}轮**: `{round_file}`\n")
                
                if success:
                    f.write(f"- **最终成功版本**: `triton_final.py` ✅\n")
                else:
                    f.write(f"- **最后尝试版本**: `triton_round{total_rounds}.py` ❌\n")
                
                f.write(f"\n## 相关文件\n\n")
                f.write(f"- **对话日志**: `conversation.log`\n")
                f.write(f"- **评估日志**: 查看对应的eval日志文件\n")
                f.write(f"- **存档版本**: `triton.py`\n")
            
            self.logger.info(f"代码版本摘要已创建: {summary_path}")
            
        except Exception as e:
            self.logger.warning(f"创建代码版本摘要失败: {e}")
    
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
        timestamp_log_dir, conversation_log_path = self.setup_logging_directories(test_dir, time_str)
        
        # 读取CUDA代码
        cuda_file_path = f"{test_dir}/cuda_ref.cu"
        try:
            with open(cuda_file_path, "r") as f:
                cuda_code = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"CUDA参考文件未找到: {cuda_file_path}")
        
        # 创建聊天树
        system_prompt = 'You are a professional GPU computing optimization expert, proficient in CUDA and Triton programming.'
        chat_tree = self.create_chat_tree(system_prompt)
        
        # 第一轮：初始生成请求
        initial_prompt = prompt.simple_initial_prompt.format(cuda_code=cuda_code)
        chat_tree.add_user_message(initial_prompt)
        
        # 记录初始prompt
        self.write_conversation_log(chat_tree, conversation_log_path, test_dir, "初始prompt已生成")
        
        generated_code = ""
        
        # 多轮对话循环
        for iteration in range(self.max_iterations):
            self.logger.info(f"第 {iteration + 1} 轮生成...")
            
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
                self.logger.error(f"第{iteration + 1}轮生成失败: {e}")
                continue
            
            # 更新对话日志
            self.write_conversation_log(
                chat_tree, conversation_log_path, test_dir, 
                f"第{iteration + 1}轮 - 模型已回复"
            )
            
            # 提取代码
            generated_code = self.extract_code_from_response(response)
            
            # 保存当前轮次的代码
            self.save_code_version(generated_code, timestamp_log_dir, iteration + 1)
            
            # 测试代码
            self.logger.info("正在测试生成的代码...")
            test_result = await self.test_triton_kernel(
                test_dir, time_str, iteration + 1, timestamp_log_dir
            )
            
            if test_result.get("success", False):
                self.logger.info("✅ 代码测试成功！")
                
                # 保存最终成功版本
                self.save_code_version(generated_code, timestamp_log_dir, iteration + 1, is_final=True)
                
                # 打印性能对比结果
                if "performance" in test_result:
                    self.logger.info(f"性能对比结果:")
                    perf_info = test_result["performance"]
                    if isinstance(perf_info, dict):
                        for key, value in perf_info.items():
                            self.logger.info(f"  {key}: {value}")
                
                # 记录成功结果到日志
                success_info = f"第{iteration + 1}轮 - 代码测试成功！\n"
                if "performance" in test_result:
                    success_info += f"性能结果: {test_result['performance']}"
                self.write_conversation_log(chat_tree, conversation_log_path, test_dir, success_info)
                
                # 创建代码版本摘要
                self.create_code_version_summary(timestamp_log_dir, iteration + 1, success=True)
                
                return generated_code
            else:
                self.logger.info(f"❌ 代码测试失败: {test_result.get('error', '未知错误')}")
                
                if iteration == self.max_iterations - 1:
                    self.logger.info(f"已达到最大迭代次数 ({self.max_iterations})，停止尝试")
                    break
                
                # 准备反馈信息
                error_info = test_result.get('error', '未知错误')
                traceback_info = test_result.get('traceback', '')
                
                feedback_prompt = prompt.feedback_prompt.format(
                    error_info=error_info, 
                    traceback_info=traceback_info
                )
                
                chat_tree.add_user_message(feedback_prompt)
                
                # 记录反馈信息到日志
                error_info_log = f"第{iteration + 1}轮 - 代码测试失败，已提供错误反馈\n错误: {error_info}"
                self.write_conversation_log(chat_tree, conversation_log_path, test_dir, error_info_log)
        
        # 如果所有迭代都失败了，仍然保存最后一次的代码
        if generated_code:
            last_attempt_path = f"{timestamp_log_dir}/triton_last_attempt.py"
            with open(last_attempt_path, "w") as f:
                f.write(generated_code)
            with open(f"{timestamp_log_dir}/triton.py", "w") as f:
                f.write(generated_code)
            self.logger.info(f"最后尝试的代码已保存到: {last_attempt_path}")
        
        # 记录最终失败日志
        final_log = f"所有{self.max_iterations}轮迭代都失败了，保存最后一次生成的代码"
        self.write_conversation_log(chat_tree, conversation_log_path, test_dir, final_log)
        
        # 创建代码版本摘要
        self.create_code_version_summary(timestamp_log_dir, self.max_iterations, success=False)
        
        return generated_code
    
    async def generate_triton_kernel(
        self, 
        test_dir: str, 
        time_str: Optional[str] = None
    ) -> str:
        """生成Triton内核代码（入口函数）
        
        Args:
            test_dir: 测试目录
            time_str: 时间戳字符串
            
        Returns:
            生成的Triton代码
        """
        try:
            return await self.generate_triton_kernel_with_feedback(test_dir, time_str)
        finally:
            await self.close()


# 向后兼容的函数接口
async def generate_triton_kernel(
    test_dir: str, 
    model_name: str = "gemini-2.5-pro", 
    time_str: Optional[str] = None,
    platform_type: PlatformType = PlatformType.GOOGLE_OFFICIAL
) -> str:
    """生成Triton内核代码（向后兼容的异步函数）
    
    Args:
        test_dir: 测试目录
        model_name: 模型名称
        time_str: 时间戳字符串
        platform_type: LLM平台类型
        
    Returns:
        生成的Triton代码
    """
    generator = TritonCodeGenerator(
        platform_type=platform_type,
        model_name=model_name
    )
    
    return await generator.generate_triton_kernel(test_dir, time_str)


async def batch_generate_kernels(levels: List[str] = None):
    """批量生成内核代码
    
    Args:
        levels: 要处理的级别列表，默认为['level1']
    """
    if levels is None:
        levels = ['level1']
    
    generator = TritonCodeGenerator()
    await generator.initialize()
    
    try:
        for level in levels:
            output_dir = f"outputs/kernelbench_c/{level}"
            if not os.path.exists(output_dir):
                print(f"目录不存在: {output_dir}")
                continue
                
            all_dirs = sorted(os.listdir(output_dir), key=lambda x: int(x.split('_')[0]))
            
            for dir_name in all_dirs:
                test_dir = f"{output_dir}/{dir_name}"
                time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                try:
                    print(f"正在为 {test_dir} 生成Triton内核")
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
                    
                except Exception as e:
                    import traceback
                    print(f"处理 {test_dir} 时出错: {e}")
                    print(traceback.format_exc())
                    continue
    finally:
        await generator.close()


# 主函数示例
async def main():
    """主函数示例"""
    # 示例1: 单个内核生成
    test_dir = "outputs/kernelbench_c/level1/1_Square_matrix_multiplication_"
    if os.path.exists(test_dir):
        try:
            generator = TritonCodeGenerator(
                platform_type=PlatformType.GOOGLE_OFFICIAL,
                model_name="gemini-2.5-pro"
            )
            await generator.initialize()
            
            code = await generator.generate_triton_kernel_with_feedback(test_dir)
            print("✅ 代码生成完成")
            print(f"生成的代码长度: {len(code)} 字符")
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
    
    # 示例2: 批量生成（取消注释以启用）
    # await batch_generate_kernels(['level1'])


if __name__ == "__main__":
    asyncio.run(main())
