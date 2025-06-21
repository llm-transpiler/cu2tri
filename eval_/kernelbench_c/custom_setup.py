from typing import Optional, Tuple
import logging
import os
from datetime import datetime

def setup_logging(
    log_file: str, 
    latest_log_file: Optional[str] = None,
    logfile_prefix: str = "",
    logger: logging.Logger = None
) -> logging.Logger:
    """
    设置日志记录
    
    Args:
        log_file: 主日志文件路径
        latest_log_file: 最新日志文件路径（可选）
        logfile_prefix: 日志文件前缀
        logger: 现有的logger对象（可选）
        
    Returns:
        配置好的logger对象
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 清除之前的处理器
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 添加主日志处理器
    main_handler = logging.FileHandler(log_file, mode='a')
    logger.addHandler(main_handler)
    
    # 如果需要，添加最新日志副本处理器
    if latest_log_file:
        os.makedirs(os.path.dirname(latest_log_file), exist_ok=True)
        latest_handler = logging.FileHandler(latest_log_file, mode='w')
        logger.addHandler(latest_handler)
    
    logger.setLevel(logging.DEBUG)
    
    # 添加轮次分隔符
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    round_info = ""
    if logfile_prefix.startswith("round"):
        round_info = f" - {logfile_prefix.replace('_', '').upper()}"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Test start time: {current_time}{round_info}")
    logger.info(f"{'='*60}")
    
    return logger


def determine_triton_file_path(
    timestamp_log_dir: Optional[str], 
    logfile_prefix: str
) -> str:
    """
    动态确定triton文件路径
    
    Args:
        timestamp_log_dir: 时间戳日志目录
        logfile_prefix: 日志文件前缀
        
    Returns:
        Triton文件路径
        
    Raises:
        ValueError: 当参数不足时
    """
    if not timestamp_log_dir:
        raise ValueError("timestamp_log_dir must be provided")
    
    if logfile_prefix.startswith("round"):
        # 多轮生成中的测试，使用对应轮次的文件
        round_num = logfile_prefix.replace("round", "").replace("_", "")
        return f"{timestamp_log_dir}/triton_round{round_num}.py"
    else:
        # 使用时间戳目录中的triton.py（最终版本）
        return f"{timestamp_log_dir}/triton.py"


def setup_log_paths(
    base_dir: str,
    model_name: str,
    time_str: Optional[str] = None,
    timestamp_log_dir: Optional[str] = None,
    logfile_prefix: str = ""
) -> Tuple[str, Optional[str]]:
    """
    设置日志路径
    
    Args:
        base_dir: 基础目录
        model_name: 模型名称
        time_str: 时间字符串（可选）
        timestamp_log_dir: 时间戳日志目录（可选）
        logfile_prefix: 日志文件前缀
        
    Returns:
        (主日志文件路径, 最新日志文件路径)
    """
    if time_str is None:
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 根据模型名创建子文件夹
    model_name_clean = model_name.replace('.', '_').replace('/', '_').replace('-', '_')
    log_dir = f"{base_dir}/logs"
    model_log_dir = f"{log_dir}/{model_name_clean}"
    
    if timestamp_log_dir is not None:
        log_file = f"{timestamp_log_dir}/{logfile_prefix}eval.log"
        # 只为最终的eval（不是round级别的）在外面保留最新副本
        if not logfile_prefix.startswith("round"):
            latest_eval_file = f"{model_log_dir}/eval_latest.log"
        else:
            latest_eval_file = None
    else:
        os.makedirs(model_log_dir, exist_ok=True)
        log_file = f"{model_log_dir}/{logfile_prefix}eval_{time_str}.log"
        latest_eval_file = None
    
    return log_file, latest_eval_file


def log_test_completion(logger: logging.Logger, success: bool = True):
    """
    记录测试完成信息
    
    Args:
        logger: 日志记录器
        success: 是否成功
    """
    status = "✅ Test completed!" if success else "❌ Test failed!"
    logger.info(f"\n{status}")
    logger.info(f"Test end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*60}\n")
