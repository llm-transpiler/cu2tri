# -*- coding: utf-8 -*-
"""
CUDA性能测试模块

该模块用于对CUDA内核进行性能测试，与PyTorch进行对比分析。
删除了所有LLM调用和Triton代码生成相关功能。
"""

import os
import sys
import asyncio
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# 导入项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

MAX_RETRIES = 3
DEFAULT_TIMEOUT = 300

# 导入CUDA评估器
try:
    from cu2tri.cuda_check.eval_cuda import CUDAKernelEvaluator
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保安装了必要的依赖包并正确配置项目路径")
    sys.exit(1)


async def process_single_kernel(test_dir, semaphore, global_time_str: str, logger: Optional[logging.Logger] = None):
    """处理单个内核的性能测试
    
    Args:
        test_dir: 测试目录路径
        semaphore: 并发限制信号量
        global_time_str: 全局时间戳，用于标识整个批次
        logger: 日志记录器
    
    Returns:
        处理结果字符串
    """
    async with semaphore:  # 限制并发数量
        # 使用全局时间戳 + 任务特定标识符
        task_time_str = f"{global_time_str}_{Path(test_dir).name}"
        
        # 简化的任务名称（只取前几个字符和数字）
        task_name = Path(test_dir).name
        # 提取任务编号（例如从 "2_Standard_matrix_multiplication_" 提取 "2"）
        task_num = task_name.split('_')[0] if '_' in task_name else task_name[:10]
        simple_task_name = f"Task_{task_num}"
        
        # 为每个任务创建独立的evaluator实例
        evaluator = CUDAKernelEvaluator()
        
        try:
            # 设置任务专用的日志记录器，避免日志竞争
            task_logger = logging.getLogger(f"{simple_task_name}")
            task_logger.setLevel(logging.INFO)
            
            # 清除之前的处理器
            for handler in task_logger.handlers[:]:
                task_logger.removeHandler(handler)
            
            # 设置控制台处理器
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                f'[{simple_task_name:8s}] | %(levelname)-5s | %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            task_logger.addHandler(console_handler)
            
            # 设置任务专用日志文件
            task_log_file = Path(test_dir) / "logs" / f"task_{task_time_str}.log"
            task_log_file = task_log_file.resolve()
            os.makedirs(task_log_file.parent, exist_ok=True)
            file_handler = logging.FileHandler(task_log_file, mode='w')
            file_formatter = logging.Formatter(
                f'[{simple_task_name:8s}] | %(levelname)-5s | %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            task_logger.addHandler(file_handler)
            
            # 防止日志向上传播
            task_logger.propagate = False
            
            # 更新evaluator的logger
            evaluator.logger = task_logger
            
            task_logger.info(f"🚀 Start CUDA performance testing {test_dir} (Batch: {global_time_str})")
            
            # 运行CUDA性能评估
            timestamp_log_dir = Path(test_dir) / "logs" / "cuda_performance" / global_time_str
            timestamp_log_dir = timestamp_log_dir.resolve()
            
            await evaluator.evaluate_cuda_kernel(
                base_dir=Path(test_dir),
                model_name="cuda_performance",
                time_str=global_time_str,
                timestamp_log_dir=timestamp_log_dir,
                return_result=False,
                timeout=DEFAULT_TIMEOUT,
                capture_output=True
            )
            
            result_msg = f"✅ Successfully tested {Path(test_dir).name}"
            task_logger.info(result_msg)
            return result_msg
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error testing {Path(test_dir).name}: {e}"
            if hasattr(evaluator, 'logger'):
                evaluator.logger.error(error_msg)
                evaluator.logger.error(traceback.format_exc())
            raise Exception(f"{error_msg}\n{traceback.format_exc()}")
        
        finally:
            # 清理日志处理器
            try:
                if hasattr(evaluator, 'logger'):
                    for handler in evaluator.logger.handlers[:]:
                        handler.close()
                        evaluator.logger.removeHandler(handler)
            except Exception as close_error:
                if logger:
                    logger.warning(f"Error cleaning up resources for {test_dir}: {close_error}")


async def batch_test_kernels(test_dirs: List[str] = None, max_concurrent: int = 10, start_id: int = 1, specific_global_time_str: str = "cuda_check", logger: Optional[logging.Logger] = None):
    """批量测试内核性能（并发执行）
    
    Args:
        test_dirs: 要处理的级别列表，默认为['01_single_op']
        max_concurrent: 最大并发数量，默认为10
        start_id: 开始处理的id号，默认为1
        specific_global_time_str: 指定的全局时间戳
        logger: 日志记录器
    """
    if test_dirs is None:
        test_dirs = ['01_single_op']
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 生成全局时间戳，用于标识整个批次的测试
    global_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    if specific_global_time_str is not None:
        global_time_str = specific_global_time_str
    logger.info(f"📅 Global batch timestamp: {global_time_str}")
    
    # 创建并发限制信号量
    semaphore = asyncio.Semaphore(max_concurrent)
    
    try:
        for sub_dir in test_dirs:
            # 修改为新的输出目录结构
            output_dir = f"/workspace/cu2tri/outputs/cu2tri/kernelbench_c/{sub_dir}"
            if not os.path.exists(output_dir):
                logger.error(f"Directory does not exist: {output_dir}")
                continue
                
            all_dirs = sorted(os.listdir(output_dir), key=lambda x: int(x.split('_')[0]))
            
            # 根据start_id过滤目录
            filtered_dirs = [d for d in all_dirs if int(d.split('_')[0]) >= start_id]
            
            if not filtered_dirs:
                logger.warning(f"No directories found with id >= {start_id} in {sub_dir}")
                continue
            
            skipped_count = len(all_dirs) - len(filtered_dirs)
            if skipped_count > 0:
                logger.info(f"⏭️ Skipped {skipped_count} directories (id < {start_id})")
            logger.info(f"🎯 Starting from id {start_id}, processing {len(filtered_dirs)} directories")
            
            # 创建并发任务列表
            tasks = []
            for dir_name in filtered_dirs:
                test_dir = f"{output_dir}/{dir_name}"
                # 创建任务协程
                task_coro = process_single_kernel(test_dir, semaphore, global_time_str, logger)
                tasks.append(task_coro)
            
            logger.info(f"🎯 Start concurrent CUDA performance testing {sub_dir} {len(tasks)} kernels (batch: {global_time_str}, max concurrent: {max_concurrent})")
            
            # 并发执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 统计并处理结果
            success_count = 0
            error_count = 0
            
            logger.info(f"\n📊 {sub_dir} testing result:")
            for i, result in enumerate(results):
                dir_name = filtered_dirs[i]
                if isinstance(result, Exception):
                    logger.error(f"  ❌ {dir_name}: {str(result).split('\\n')[0]}")
                    error_count += 1
                else:
                    logger.info(f"  {result}")
                    success_count += 1
            
            logger.info(f"\n🏆 {sub_dir} summary: {success_count} passed, {error_count} failed")
            logger.info("-" * 60)
                    
    except Exception as e:
        logger.error(f"Batch testing error: {e}")
        import traceback
        logger.error(traceback.format_exc())


# 主函数示例
async def main():
    """主函数示例"""
    # 批量CUDA性能测试
    await batch_test_kernels(['01_single_op'], max_concurrent=1, specific_global_time_str="cuda_check", start_id=71)


if __name__ == "__main__":
    asyncio.run(main())
