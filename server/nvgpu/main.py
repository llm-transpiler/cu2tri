#!/usr/bin/env python3
"""Main entry point for NVGPU server."""
import argparse
import signal
import sys
from pathlib import Path

from server.nvgpu.config import config, GPUMode, LoadBalancingStrategy
from server.nvgpu.logger import setup_logger
from server.nvgpu.gpu_manager import GPUManager
from server.nvgpu.task_queue import TaskQueue
from server.nvgpu.task_runner import TaskRunner
from server.nvgpu.scheduler import Scheduler
from server.nvgpu.api_server import init_app, app
from server.nvgpu.gpu_config_loader import GPUConfigLoader
from server.common.timezone import format_timestamp, set_default_timezone
from server.common.timer import (
    monotonic_timestamp_ns,
    perf_counter_timestamp_ns,
)
import uvicorn

# Define NVGPU root directory (main.py's parent directory)
NVGPU_ROOT = Path(__file__).parent.resolve()

# Setup dual logging system:
# 1. Timestamped log for current session
# 2. History log (appended) for all sessions
set_default_timezone(config.timezone)
timestamp = format_timestamp()
log_file = str(NVGPU_ROOT / "logs" / f"nvgpu_server_{timestamp}.log")
history_log_file = str(NVGPU_ROOT / "logs" / "nvgpu_server.log")
logger = setup_logger("main", log_file=log_file, history_log_file=history_log_file)

logger.info(
    "Timer setup: perf_counter_ns for durations (sample=%d), monotonic_ns for ordering (sample=%d)",
    perf_counter_timestamp_ns(),
    monotonic_timestamp_ns(),
)


class NVGPUServer:
    """Main NVGPU server class."""
    
    def __init__(self, gpu_config_file: str | None = None, 
                 log_file_path: str | None = None, 
                 history_log_file_path: str | None = None):
        """Initialize server.
        
        Args:
            gpu_config_file: Path to GPU configuration YAML file
            log_file_path: Path to current session log file (timestamped)
            history_log_file_path: Path to history log file (appended)
        """
        # Store log file paths (use absolute paths)
        if log_file_path:
            self.log_file = log_file_path if Path(log_file_path).is_absolute() else str(NVGPU_ROOT / log_file_path)
        else:
            self.log_file = str(NVGPU_ROOT / "logs" / f"nvgpu_server_{format_timestamp()}.log")
        
        if history_log_file_path:
            self.history_log_file = history_log_file_path if Path(history_log_file_path).is_absolute() else str(NVGPU_ROOT / history_log_file_path)
        else:
            self.history_log_file = str(NVGPU_ROOT / "logs" / "nvgpu_server.log")

        self.logger = logger

        # Setup logging for all components
        import logging
        log_level = getattr(logging, config.log_level.upper(), logging.DEBUG)
        
        # Re-configure loggers for all components with dual logging
        import server.nvgpu.gpu_manager as gpu_manager
        import server.nvgpu.task_queue as task_queue
        import server.nvgpu.task_runner as task_runner
        import server.nvgpu.scheduler as scheduler
        import server.nvgpu.api_server as api_server
        import server.nvgpu.gpu_config_loader as gcl
        
        gpu_manager.logger = setup_logger("gpu_manager", log_file=self.log_file, 
                                          history_log_file=self.history_log_file, level=log_level)
        task_queue.logger = setup_logger("task_queue", log_file=self.log_file, 
                                         history_log_file=self.history_log_file, level=log_level)
        task_runner.logger = setup_logger("task_runner", log_file=self.log_file, 
                                          history_log_file=self.history_log_file, level=log_level)
        scheduler.logger = setup_logger("scheduler", log_file=self.log_file, 
                                        history_log_file=self.history_log_file, level=log_level)
        api_server.logger = setup_logger("api_server", log_file=self.log_file, 
                                         history_log_file=self.history_log_file, level=log_level)
        if gpu_config_file:
            gcl.logger = setup_logger("gpu_config", log_file=self.log_file, 
                                     history_log_file=self.history_log_file, level=log_level)
        
        # Load GPU configuration if provided
        self.gpu_config_loader = None
        if gpu_config_file:
            self.gpu_config_loader = GPUConfigLoader(gpu_config_file)
            if self.gpu_config_loader.load():
                # Set global config loader for other modules
                import server.nvgpu.gpu_manager as gpu_manager
                import server.nvgpu.task_runner as task_runner
                gpu_manager.gpu_config_loader = self.gpu_config_loader
                task_runner.gpu_config_loader = self.gpu_config_loader
        
        self.gpu_manager = GPUManager()
        self.task_queue = TaskQueue()
        self.task_runner = TaskRunner()
        
        # Set dependencies for advanced error handling
        self.gpu_manager.set_dependencies(self.task_queue, self.task_runner)
        
        self.scheduler = Scheduler(self.gpu_manager, self.task_queue, self.task_runner)
    
    def start(self):
        """Start all server components."""
        logger.info("=" * 60)
        logger.info("Starting NVGPU Server")
        logger.info("=" * 60)
        
        # Auto-register GPUs from config if enabled
        if self.gpu_config_loader and self.gpu_config_loader.should_auto_register():
            logger.info("Auto-registering GPUs from configuration...")
            for gpu_config in self.gpu_config_loader.get_enabled_gpus():
                from server.nvgpu.config import GPUMode
                mode = GPUMode.EXCLUSIVE if gpu_config.default_mode == "exclusive" else GPUMode.SHARED
                self.gpu_manager.register_gpu(
                    gpu_id=gpu_config.logical_id,
                    mode=mode,
                    memory_threshold=gpu_config.memory_threshold,
                    max_concurrent_tasks=gpu_config.max_concurrent_tasks
                )
                logger.info(f"  Registered GPU {gpu_config.logical_id}: {gpu_config.name} "
                            f"(mode={mode}, max_tasks={gpu_config.max_concurrent_tasks})")
        
        # Start GPU monitoring
        self.gpu_manager.start_monitoring()
        
        # Start scheduler
        self.scheduler.start()
        strategy_name = (
            config.load_balancing_strategy.value
            if config.load_balancing_strategy
            else LoadBalancingStrategy.FILL.value
        )
        logger.info(f"Load balancing strategy: {strategy_name}")
        
        # Initialize FastAPI app
        init_app(self.gpu_manager, self.task_queue, self.scheduler, self.task_runner)
        
        logger.info(f"Server ready on http://{config.host}:{config.port}")
        logger.info(f"Registered GPUs: {[g.gpu_id for g in self.gpu_manager.list_gpus()]}")
        
        # Configure uvicorn logging to use dual log files (timestamped + history)
        from pathlib import Path
        log_file_abs = str(Path(self.log_file).absolute())
        history_log_file_abs = str(Path(self.history_log_file).absolute())
        
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "server.nvgpu.logger.TimezoneFormatter",
                    "format": "%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "formatter": "default",
                    "class": "logging.FileHandler",
                    "filename": log_file_abs,
                },
                "history_file": {
                    "formatter": "default",
                    "class": "logging.FileHandler",
                    "filename": history_log_file_abs,
                    "mode": "a",  # Append mode for history log
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default", "file", "history_file"], "level": "INFO"},
                "uvicorn.error": {"handlers": ["default", "file", "history_file"], "level": "INFO", "propagate": False},
                "uvicorn.access": {"handlers": ["default", "file", "history_file"], "level": "INFO", "propagate": False},
            },
        }
        
        # Run API server (blocking)
        uvicorn.run(app, host=config.host, port=config.port, log_config=log_config)
    
    def stop(self):
        """Stop all server components."""
        logger.info("Shutting down NVGPU Server...")
        
        self.scheduler.stop()
        self.gpu_manager.shutdown()
        
        logger.info("Server stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="NVGPU Server - GPU task scheduling service")
    parser.add_argument("--host", default=config.host, help="Server host")
    parser.add_argument("--port", type=int, default=config.port, help="Server port")
    parser.add_argument("--log-file", default=None, help="Log file path (default: auto-generated with timestamp)")
    parser.add_argument("--log-level", default=config.log_level, help="Log level")
    parser.add_argument("--gpu-config", default=None,
                       help="GPU resource configuration file (YAML)")
    parser.add_argument("--gpus", type=int, nargs="+", 
                       help="GPU IDs to register at startup (overrides config file)")
    parser.add_argument("--gpu-mode", choices=["exclusive", "shared"], 
                       help="Default GPU mode (when using --gpus)")
    parser.add_argument("--memory-threshold", type=float,
                       help="Default memory threshold (when using --gpus)")
    parser.add_argument(
        "--lb-strategy",
        choices=["default"] + [s.value for s in LoadBalancingStrategy],
        default="round_robin",
        help="Load balancing strategy (default: legacy fill-first scheduling)",
    )
    
    args = parser.parse_args()
    
    # Update config
    global log_file, logger
    
    config.host = args.host
    config.port = args.port
    
    # Handle log file
    if args.log_file:
        # User specified a custom log file (no timestamp)
        log_file = args.log_file
        config.log_file = args.log_file
        logger.info(f"Using custom log file: {log_file}")
    # else: keep the auto-generated timestamped log_file
    
    config.log_level = args.log_level
    if args.gpu_mode:
        config.default_gpu_mode = GPUMode(args.gpu_mode)
    if args.memory_threshold:
        config.default_memory_threshold = args.memory_threshold
    if args.lb_strategy == "default":
        config.load_balancing_strategy = None
    else:
        strategy = LoadBalancingStrategy(args.lb_strategy)
        # Treat fill as explicit legacy mode
        config.load_balancing_strategy = (
            None if strategy == LoadBalancingStrategy.FILL else strategy
        )
    
    # Re-setup main logger with dual logging (timestamped + history)
    import logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger("main", log_file=log_file, history_log_file=history_log_file, level=log_level)
    logger.info(f"Session log: {log_file}")
    logger.info(f"History log: {history_log_file}")
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create server with GPU config and dual logging
    # Handle GPU config file path (convert to absolute if needed)
    gpu_config_file = None
    if args.gpu_config:
        config_path = Path(args.gpu_config)
        if not config_path.is_absolute():
            # Try relative to NVGPU_ROOT first, then current working directory
            nvgpu_relative = NVGPU_ROOT / config_path
            if nvgpu_relative.exists():
                gpu_config_file = str(nvgpu_relative)
            elif config_path.exists():
                gpu_config_file = str(config_path.resolve())
        elif config_path.exists():
            gpu_config_file = str(config_path)
    
    server = NVGPUServer(gpu_config_file=gpu_config_file, 
                        log_file_path=log_file, 
                        history_log_file_path=history_log_file)
    
    # Register GPUs if specified via command line (overrides config file)
    if args.gpus:
        for gpu_id in args.gpus:
            server.gpu_manager.register_gpu(
                gpu_id,
                mode=config.default_gpu_mode,
                memory_threshold=config.default_memory_threshold
            )
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user", exc_info=True)
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()

'''
root@ubuntu-ThinkStation-P520:/workspace# cd /workspace/server/nvgpu && python main.py > /tmp/nvgpu_test_server.log 2>&1 &
[1] 2511310
root@ubuntu-ThinkStation-P520:/workspace# ps aux | grep "python main.py" | grep -v grep
root     2511312  2.6  0.0 325064 66692 pts/231  Sl   04:10   0:00 python main.py
root@ubuntu-ThinkStation-P520:/workspace# pgrep -f "python main.py"
2511312
'''
'''
grep 的 -v 选项：这个选项的意思是 --invert-match，也就是反向查找。它会显示出所有不包含指定字符串的行。
这个终端不能断。ctrl+d或者关掉这个进程也会消失
'''

'''
cd /workspace/server/nvgpu && nohup python main.py > /tmp/nvgpu_server_timing_test.log 2>&1 & echo "Server PID: $!"
'''
