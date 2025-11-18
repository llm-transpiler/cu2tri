"""
通用日志模块
提供可配置的日志功能，支持不同的logger名称和输出文件
"""

import logging
import logging.handlers
import os
import shutil
from pathlib import Path
from typing import Optional, Union

class NumberedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Custom rotating file handler using log.0, log.1, log.2 naming format
    """

    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)

    def doRollover(self):
        """
        Perform log rotation using log.0, log.1, log.2 naming format
        If backup_count is -1, keep unlimited backups
        """
        # Handle no rotation case (backupCount == 0)
        if self.backupCount == 0:
            # Truncate current file and continue
            if self.stream:
                self.stream.close()
                self.stream = None
            # Truncate the current file (same as open with 'w' mode)
            with open(self.baseFilename, 'w', encoding=self.encoding):
                pass  # Truncate to 0 bytes
            # Reopen the file for continued writing
            if not self.delay:
                self.stream = open(self.baseFilename, 'a', encoding=self.encoding)
            return

        # For rotation cases, close current stream first
        if self.stream:
            self.stream.close()
            self.stream = None

        # Find the highest existing log number
        max_existing = -1
        if self.backupCount > 0:
            # Limited backup mode: find max up to backupCount-1
            for i in range(self.backupCount - 1, -1, -1):
                if os.path.exists(self.baseFilename + f".{i}"):
                    max_existing = i
                    break
        else:  # backupCount < 0
            # Unlimited backup mode: find all existing files
            i = 0
            while True:
                if os.path.exists(self.baseFilename + f".{i}"):
                    max_existing = i
                    i += 1
                else:
                    break

        # Rotate files: log.N -> log.N+1
        if self.backupCount > 0:
            # Limited backup: only keep up to backupCount files
            for i in range(min(max_existing, self.backupCount - 2), -1, -1):
                src = self.baseFilename + f".{i}"
                dst = self.baseFilename + f".{i + 1}"

                if os.path.exists(src):
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.rename(src, dst)
        else:
            # Unlimited backup: rotate all existing files
            for i in range(max_existing, -1, -1):
                src = self.baseFilename + f".{i}"
                dst = self.baseFilename + f".{i + 1}"

                if os.path.exists(src):
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.rename(src, dst)

        # Rename current log file to log.0
        dst = self.baseFilename + ".0"
        if os.path.exists(dst):
            os.remove(dst)
        self.rotate(self.baseFilename, dst)

        # Create new log file
        if not self.delay:
            self.stream = open(self.baseFilename, 'w', encoding=self.encoding)

    @staticmethod
    def merge_logs(log_file_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Merge all rotated log files into one large file

        Args:
            log_file_path: Base log file path (e.g., /path/to/app.log)
            output_path: Output file path for merged file, if None uses original path+_merged

        Returns:
            Path to the merged file
        """
        log_file_path = Path(log_file_path)

        if output_path is None:
            output_path = log_file_path.parent / f"{log_file_path.stem}_merged{log_file_path.suffix}"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Find all rotated files (unlimited backup support)
        log_files = []
        i = 0
        while True:
            rotated_file = log_file_path.parent / f"{log_file_path.name}.{i}"
            if rotated_file.exists():
                log_files.append(rotated_file)
                i += 1
            else:
                break

        # Order by time (oldest first)
        log_files.reverse()

        # Add main log file to end if it exists
        if log_file_path.exists():
            log_files.append(log_file_path)

        # Merge files
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for log_file in log_files:
                try:
                    with open(log_file, 'r', encoding='utf-8') as infile:
                        shutil.copyfileobj(infile, outfile)
                except Exception as e:
                    # If a file cannot be read, log error and continue
                    outfile.write(f"\n\n=== Error: Unable to read {log_file.name}: {str(e)} ===\n\n")

        return str(output_path)


def create_numbered_rotating_handler(
    log_file: Union[str, Path],
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = -1,  # -1 means unlimited backups
    encoding: str = 'utf-8',
    delay: bool = False
) -> NumberedRotatingFileHandler:
    """
    Create numbered rotating file handler

    Args:
        log_file: Log file path
        max_bytes: Maximum file size (bytes)
        backup_count: Number of backup files (-1 for unlimited)
        encoding: File encoding
        delay: Whether to delay file creation

    Returns:
        NumberedRotatingFileHandler instance
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    return NumberedRotatingFileHandler(
        filename=str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding=encoding,
        delay=delay
    )


def configure_logger(
    logger_name: str,
    log_file: Optional[Union[str, Path]] = None,
    log_level: str = "INFO",
    console_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = -1,  # -1 means unlimited backups
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure a general-purpose logger

    Args:
        logger_name: Logger name
        log_file: Log file path, if None only console output
        log_level: Log level
        console_output: Whether to output to console
        max_bytes: Maximum log file size in bytes
        backup_count: Number of log file backups to retain (-1 for unlimited)
        log_format: Custom log format, if None uses default format

    Returns:
        logging.Logger: Configured logger
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 获取logger
    logger = logging.getLogger(logger_name)

    # 清除已有的处理器，避免重复输出
    logger.handlers.clear()

    # 设置日志级别
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logger.setLevel(log_level_map.get(log_level.upper(), logging.INFO))

    # 创建格式化器
    formatter = logging.Formatter(log_format)

    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File output
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Use custom numbered rotating handler for log rotation with log.0, log.1, log.2 format
        file_handler = NumberedRotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(
    logger_name: str,
    log_file: Optional[Union[str, Path]] = None,
    log_level: str = "INFO",
    console_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = -1,  # -1 means unlimited backups
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Get or create a logger

    This function is a simplified version of configure_logger. If logger already exists, return existing logger

    Args:
        logger_name: Logger name
        log_file: Log file path, if None only console output
        log_level: Log level
        console_output: Whether to output to console
        max_bytes: Maximum log file size in bytes
        backup_count: Number of log file backups to retain (-1 for unlimited)
        log_format: Custom log format, if None uses default format

    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(logger_name)

    # 如果logger已经配置过（有处理器），直接返回
    if logger.handlers:
        return logger

    # 否则配置新的logger
    return configure_logger(
        logger_name=logger_name,
        log_file=log_file,
        log_level=log_level,
        console_output=console_output,
        max_bytes=max_bytes,
        backup_count=backup_count,
        log_format=log_format
    )


# 预定义的便捷函数
def get_llm_trans_logger(
    log_file: Optional[Union[str, Path]] = None,
    log_level: str = "INFO"
) -> logging.Logger:
    """获取llm_trans专用的日志记录器"""
    return get_logger(
        logger_name="llm_trans",
        log_file=log_file,
        log_level=log_level
    )


def get_cu2tri_logger(
    log_file: Optional[Union[str, Path]] = None,
    log_level: str = "INFO"
) -> logging.Logger:
    """获取cu2tri项目专用的日志记录器"""
    return get_logger(
        logger_name="cu2tri",
        log_file=log_file,
        log_level=log_level
    )