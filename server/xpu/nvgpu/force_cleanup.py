#!/usr/bin/env python3
# 感觉不是很能用
"""
增强版强制清理工具
彻底清理卡住的GPU进程和资源
"""

import os
import sys
import time
import signal
import subprocess
import logging
from pathlib import Path

def setup_logger():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("ForceCleanup")

def get_process_tree(pid):
    """获取进程树（包括子进程）"""
    try:
        result = subprocess.run(
            ['pstree', '-p', str(pid)], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return None
    except:
        return None

def should_exclude_process(pid):
    """检查是否应该排除某个进程"""
    try:
        result = subprocess.run(
            ['ps', '-p', str(pid), '-o', 'cmd', '--no-headers'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            cmd = result.stdout.strip()
            
            # 排除的进程模式
            exclude_patterns = [
                # 'nnsmith',          # nnsmith 工具
                # 'miniconda3',       # conda环境（如果是长期运行的其他任务）
                'tmux',             # tmux会话
                'vim',              # 编辑器
                'nano',             # 编辑器
                'ssh',              # SSH连接
            ]
            
            for pattern in exclude_patterns:
                if pattern in cmd.lower():
                    return True
                    
            return False
    except:
        return False

def find_gpu_processes():
    """查找所有GPU相关进程"""
    logger = logging.getLogger("ForceCleanup")
    gpu_processes = []
    
    # 获取当前进程PID，避免杀死自己
    current_pid = os.getpid()
    
    # 查找包含gpu、cuda、triton等关键词的进程，但重点关注我们的服务器进程
    keywords = ['start_server', 'api_server', 'gpu_manager', 'eval_triton', 'gemini_call']
    
    for keyword in keywords:
        try:
            result = subprocess.run(
                ['pgrep', '-f', keyword], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                pids = [int(pid.strip()) for pid in result.stdout.split() if pid.strip()]
                # 过滤掉自己的PID和需要排除的进程
                pids = [pid for pid in pids if pid != current_pid and not should_exclude_process(pid)]
                gpu_processes.extend(pids)
                logger.info(f"Found {len(pids)} processes for keyword '{keyword}': {pids}")
        except Exception as e:
            logger.warning(f"Error searching for keyword '{keyword}': {e}")
    
    # 去重并再次确保当前进程不在列表中
    gpu_processes = list(set(gpu_processes))
    gpu_processes = [pid for pid in gpu_processes if pid != current_pid]
    
    return gpu_processes

def get_process_info(pid):
    """获取进程详细信息"""
    try:
        result = subprocess.run(
            ['ps', '-p', str(pid), '-o', 'pid,ppid,cmd,state,time'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                return lines[1]  # 跳过标题行
        return None
    except:
        return None

def kill_process_forcefully(pid, logger):
    """强制杀死进程"""
    try:
        # 首先尝试SIGTERM
        logger.info(f"Sending SIGTERM to process {pid}")
        os.kill(pid, signal.SIGTERM)
        
        # 等待2秒
        time.sleep(2)
        
        # 检查进程是否还存在
        try:
            os.kill(pid, 0)  # 发送信号0检查进程是否存在
            logger.warning(f"Process {pid} still alive after SIGTERM, sending SIGKILL")
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
            
            # 再次检查
            try:
                os.kill(pid, 0)
                logger.error(f"Process {pid} still alive after SIGKILL!")
                return False
            except ProcessLookupError:
                logger.info(f"Process {pid} successfully killed with SIGKILL")
                return True
                
        except ProcessLookupError:
            logger.info(f"Process {pid} successfully terminated with SIGTERM")
            return True
            
    except ProcessLookupError:
        logger.info(f"Process {pid} already terminated")
        return True
    except PermissionError:
        logger.error(f"Permission denied to kill process {pid}")
        return False
    except Exception as e:
        logger.error(f"Error killing process {pid}: {e}")
        return False

def cleanup_cuda_resources():
    """清理CUDA资源"""
    logger = logging.getLogger("ForceCleanup")
    
    try:
        # 使用nvidia-smi重置GPU
        logger.info("Attempting to reset GPU state...")
        result = subprocess.run(['nvidia-smi', '--gpu-reset'], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info("GPU reset successful")
        else:
            logger.warning(f"GPU reset failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"Error resetting GPU: {e}")
    
    try:
        # 清理共享内存
        logger.info("Cleaning shared memory...")
        result = subprocess.run(['ipcs', '-m'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'torch' in line.lower() or 'cuda' in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        shmid = parts[1]
                        try:
                            subprocess.run(['ipcrm', '-m', shmid], timeout=5)
                            logger.info(f"Removed shared memory segment {shmid}")
                        except:
                            pass
    except Exception as e:
        logger.warning(f"Error cleaning shared memory: {e}")

def cleanup_multiprocessing_resources():
    """清理multiprocessing资源"""
    logger = logging.getLogger("ForceCleanup")
    
    try:
        # 查找并清理multiprocessing相关的临时文件
        import tempfile
        temp_dir = Path(tempfile.gettempdir())
        
        # 查找multiprocessing相关的临时文件，但排除CUDA编译器临时文件
        patterns = ['pymp-*', '*multiprocessing*']
        
        # 对于以tmp开头的文件，需要更精确的过滤
        for temp_file in temp_dir.glob('tmp*'):
            try:
                # 排除CUDA编译器临时文件 (tmpxft_* 系列)
                if temp_file.name.startswith('tmpxft_'):
                    continue
                    
                # 排除其他可能重要的临时文件
                if any(keyword in temp_file.name.lower() for keyword in ['cuda', 'nvcc', 'ptx', 'cubin']):
                    logger.debug(f"Skipping CUDA-related temp file: {temp_file}")
                    continue
                
                # 只清理明确的multiprocessing相关临时文件
                if any(keyword in temp_file.name.lower() for keyword in ['pymp', 'multiprocessing', 'mp_']):
                    if temp_file.is_file():
                        temp_file.unlink()
                        logger.info(f"Removed multiprocessing temp file: {temp_file}")
                    elif temp_file.is_dir():
                        import shutil
                        shutil.rmtree(temp_file)
                        logger.info(f"Removed multiprocessing temp dir: {temp_file}")
                else:
                    logger.debug(f"Skipping non-multiprocessing temp file: {temp_file}")
                    
            except Exception as e:
                logger.warning(f"Error processing {temp_file}: {e}")
        
        # 处理其他明确的multiprocessing模式
        for pattern in patterns:
            for temp_file in temp_dir.glob(pattern):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                        logger.info(f"Removed temp file: {temp_file}")
                    elif temp_file.is_dir():
                        import shutil
                        shutil.rmtree(temp_file)
                        logger.info(f"Removed temp dir: {temp_file}")
                except Exception as e:
                    logger.warning(f"Error removing {temp_file}: {e}")
                    
    except Exception as e:
        logger.warning(f"Error cleaning multiprocessing resources: {e}")

def main():
    """主函数"""
    logger = setup_logger()
    
    logger.info("🧹 Starting enhanced force cleanup...")
    
    # 1. 查找GPU相关进程
    logger.info("Step 1: Finding GPU-related processes...")
    gpu_processes = find_gpu_processes()
    
    if not gpu_processes:
        logger.info("No GPU processes found")
    else:
        logger.info(f"Found {len(gpu_processes)} GPU-related processes")
        
        # 显示进程信息
        for pid in gpu_processes:
            info = get_process_info(pid)
            if info:
                logger.info(f"  Process {pid}: {info}")
                tree = get_process_tree(pid)
                if tree:
                    logger.info(f"  Process tree:\n{tree}")
    
    # 2. 询问用户是否继续
    if gpu_processes:
        response = input(f"\nFound {len(gpu_processes)} processes to kill. Continue? (y/N): ")
        if response.lower() != 'y':
            logger.info("Cleanup cancelled by user")
            return
    
    # 3. 强制终止进程
    if gpu_processes:
        logger.info("Step 2: Force killing processes...")
        killed_count = 0
        failed_count = 0
        
        # 按PID降序排列，先杀父进程
        gpu_processes.sort(reverse=True)
        
        for pid in gpu_processes:
            if kill_process_forcefully(pid, logger):
                killed_count += 1
            else:
                failed_count += 1
        
        logger.info(f"Process cleanup complete: {killed_count} killed, {failed_count} failed")
    
    # 4. 清理CUDA资源
    logger.info("Step 3: Cleaning CUDA resources...")
    cleanup_cuda_resources()
    
    # 5. 清理multiprocessing资源
    logger.info("Step 4: Cleaning multiprocessing resources...")
    cleanup_multiprocessing_resources()
    
    # 6. 最终验证
    logger.info("Step 5: Final verification...")
    remaining_processes = find_gpu_processes()
    if remaining_processes:
        logger.warning(f"Warning: {len(remaining_processes)} processes still running: {remaining_processes}")
    else:
        logger.info("✅ All GPU processes cleaned successfully")
    
    logger.info("🧹 Enhanced force cleanup completed!")

if __name__ == "__main__":
    main() 