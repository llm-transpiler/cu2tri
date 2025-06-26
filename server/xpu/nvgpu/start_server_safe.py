#!/usr/bin/env python3
"""
改进的GPU服务器启动脚本
包含更好的信号处理和资源清理机制
"""

import os
import sys
import signal
import logging
import asyncio
import multiprocessing
import time
import threading
from pathlib import Path

# 强制设置多进程启动方法
multiprocessing.set_start_method('spawn', force=True)

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from server.xpu.nvgpu.api_server import GPUAPIServer
import uvicorn


class SafeGPUServer:
    """安全的GPU服务器实现"""
    
    def __init__(self):
        self.api_server = None
        self.server = None
        self.logger = self._setup_logger()
        self.shutdown_timeout = 30  # 30秒关闭超时
        self.shutdown_requested = False
        self.shutdown_lock = threading.Lock()
        
    def _setup_logger(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger("SafeGPUServer")
        return logger
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            # 使用锁防止重复处理
            with self.shutdown_lock:
                if self.shutdown_requested:
                    self.logger.warning(f"Shutdown already requested, ignoring signal {signum}")
                    return
                self.shutdown_requested = True
            
            signal_name = signal.Signals(signum).name
            self.logger.warning(f"Received {signal_name} ({signum}), initiating graceful shutdown...")
            
            # 在新线程中处理关闭，避免阻塞信号处理器
            shutdown_thread = threading.Thread(
                target=self._handle_shutdown_in_thread,
                daemon=True
            )
            shutdown_thread.start()
        
        def critical_signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            self.logger.error(f"Received critical signal {signal_name} ({signum}), forcing immediate exit")
            self._force_exit()
        
        # 注册优雅关闭信号
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # 注册强制退出信号
        signal.signal(signal.SIGUSR1, critical_signal_handler)
        
        self.logger.info("Signal handlers registered")
    
    def _handle_shutdown_in_thread(self):
        """在独立线程中处理关闭逻辑"""
        try:
            # 如果有运行中的事件循环，在其中调度关闭任务
            try:
                loop = asyncio.get_running_loop()
                # 创建异步任务来处理关闭
                future = asyncio.run_coroutine_threadsafe(self._graceful_shutdown(), loop)
                # 等待关闭完成，但设置超时
                future.result(timeout=self.shutdown_timeout)
            except Exception as e:
                self.logger.error(f"Error during graceful shutdown: {e}")
                self._force_exit()
        except Exception as e:
            self.logger.error(f"Error in shutdown thread: {e}")
            self._force_exit()
    
    async def _graceful_shutdown(self):
        """优雅关闭"""
        try:
            self.logger.info("Starting graceful shutdown...")
            
            # 设置关闭超时
            shutdown_start = time.time()
            
            # 1. 停止Uvicorn服务器（停止接受新请求）
            if self.server:
                self.logger.info("Stopping Uvicorn server...")
                self.server.should_exit = True
                # 等待服务器关闭
                timeout_remaining = self.shutdown_timeout - (time.time() - shutdown_start)
                if timeout_remaining > 0:
                    await asyncio.sleep(min(2, timeout_remaining))  # 给服务器时间关闭
            
            # 2. 停止GPU管理器（停止接受新任务）
            if self.api_server and self.api_server.gpu_manager:
                self.logger.info("Stopping GPU manager...")
                await self.api_server.gpu_manager.stop()
                self.logger.info("GPU manager stopped")
            
            # 3. 等待正在运行的任务完成（有超时）
            if self.api_server and self.api_server.gpu_manager:
                timeout_remaining = self.shutdown_timeout - (time.time() - shutdown_start)
                if timeout_remaining > 5:  # 至少给5秒来等待任务
                    self.logger.info(f"Waiting for running tasks to complete (timeout: {timeout_remaining:.1f}s)...")
                    await self._wait_for_tasks_completion(min(timeout_remaining - 2, 15))  # 留2秒给最后清理
            
            # 4. 清理资源
            self.logger.info("Cleaning up resources...")
            self._cleanup_resources()
            
            self.logger.info("Graceful shutdown completed")
            
            # 给一点时间让日志输出完成
            await asyncio.sleep(0.5)
            
        except Exception as e:
            self.logger.error(f"Error during graceful shutdown: {e}")
        finally:
            # 确保进程退出
            os._exit(0)
    
    async def _wait_for_tasks_completion(self, timeout: float):
        """等待任务完成"""
        start_time = time.time()
        last_count = -1
        
        while time.time() - start_time < timeout:
            try:
                if self.api_server and self.api_server.gpu_manager:
                    queue_status = await self.api_server.gpu_manager.get_queue_status()
                    running_count = queue_status.get('running_tasks_count', 0)
                    
                    if running_count == 0:
                        self.logger.info("All tasks completed")
                        return
                    
                    # 只在任务数量变化时打印日志
                    if running_count != last_count:
                        self.logger.info(f"Waiting for {running_count} running tasks...")
                        last_count = running_count
                    
                    await asyncio.sleep(2)  # 每2秒检查一次
                else:
                    return
                    
            except Exception as e:
                self.logger.error(f"Error checking task status: {e}")
                break
        
        if last_count > 0:
            self.logger.warning(f"Timeout waiting for {last_count} tasks completion after {timeout}s")
    
    def _cleanup_resources(self):
        """清理资源"""
        try:
            # 清理多进程资源
            try:
                # 强制清理multiprocessing资源
                import multiprocessing.util
                multiprocessing.util._exit_function()
            except Exception as e:
                self.logger.warning(f"Error cleaning multiprocessing resources: {e}")
            
            # 清理CUDA资源
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                self.logger.warning(f"Error cleaning CUDA resources: {e}")
            
            self.logger.info("Resource cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")
    
    def _force_exit(self):
        """强制退出"""
        self.logger.error("Performing force exit...")
        try:
            self._cleanup_resources()
        except:
            pass
        finally:
            os._exit(1)
    
    async def start_server(self, host="0.0.0.0", port=8000):
        """启动服务器"""
        try:
            # 设置信号处理器
            self._setup_signal_handlers()
            
            # 创建日志目录
            log_dir = Path(__file__).parent / "logs"
            log_dir.mkdir(exist_ok=True)
            
            # 创建API服务器
            self.logger.info("Initializing API server...")
            self.api_server = GPUAPIServer(
                host=host,
                port=port,
                log_dir=str(log_dir),
                logger=self.logger
            )
            
            self.logger.info("API server initialized")
            
            # 启动Web服务器
            config = uvicorn.Config(
                self.api_server.app,
                host=host,
                port=port,
                log_level="info",
                loop="asyncio"
            )
            
            server = uvicorn.Server(config)
            self.server = server
            
            self.logger.info(f"Starting GPU resource server on {host}:{port}")
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Error starting server: {e}")
            raise
    
    def run(self, host="0.0.0.0", port=8000):
        """运行服务器"""
        try:
            asyncio.run(self.start_server(host, port))
        except KeyboardInterrupt:
            self.logger.info("Server interrupted by user")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            self._force_exit()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Safe GPU Resource Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind to")
    
    args = parser.parse_args()
    
    server = SafeGPUServer()
    server.run(args.host, args.port)


if __name__ == "__main__":
    main() 