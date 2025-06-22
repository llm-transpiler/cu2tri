#!/usr/bin/env python3
"""
GPU Management Server Startup Script

Provides easy startup for the GPU management API server with different configurations.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from utils.set_env import PROJECT_ROOT

from server.xpu.nvgpu.api_server import GPUAPIServer
import uvicorn


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(levelname)s] - %(message)s'
    )
    
    # Setup logger for this module only
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        print(f"Logging to file: {log_file}")
    
    # Don't propagate to avoid duplicate logs
    logger.propagate = False
    
    return logger


def parse_gpu_ids(gpu_str: str) -> list:
    """Parse GPU IDs from comma-separated string"""
    if not gpu_str:
        return []
    
    try:
        return [int(gpu_id.strip()) for gpu_id in gpu_str.split(',')]
    except ValueError:
        raise ValueError(f"Invalid GPU IDs format: {gpu_str}")


async def main():
    """Async main function"""
    # Parse arguments and setup logging first
    parser = argparse.ArgumentParser(
        description="GPU Management API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with default settings
  python start_server.py
  
  # Start with specific GPUs visible
  python start_server.py --gpus 0,1,2,3
  
  # Start with custom port and logging
  python start_server.py --port 8090 --log-level DEBUG --log-file server.log
  
  # Start with H100 GPUs only (if you know their IDs)
  python start_server.py --gpus 0,1,2,3 --refresh-interval 5
        """
    )
    
    # Server configuration
    parser.add_argument(
        '--host', 
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8081,
        help='Port to bind to (default: 8081)'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=1,
        help='Number of worker processes (default: 1)'
    )
    
    # GPU configuration
    parser.add_argument(
        '--gpus',
        type=str,
        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2,3")'
    )
    parser.add_argument(
        '--refresh-interval',
        type=int,
        default=10,
        help='GPU status refresh interval in seconds (default: 10)'
    )
    parser.add_argument(
        '--max-wait-time',
        type=int,
        default=30,
        help='Maximum task wait time in minutes (default: 30)'
    )
    
    # Logging configuration
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (optional)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default=PROJECT_ROOT / 'server' / 'logs' / 'xpu' / 'nvgpu',
        help='Log directory for GPU tasks (default: server/logs/xpu/nvgpu)'
    )
    
    # Development options
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    # Set CUDA_VISIBLE_DEVICES if specified
    if args.gpus:
        try:
            gpu_ids = parse_gpu_ids(args.gpus)
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
            logger.info(f"Set CUDA_VISIBLE_DEVICES to: {gpu_ids}")
        except ValueError as e:
            logger.error(f"Error parsing GPU IDs: {e}")
            sys.exit(1)
    
    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"GPU task logs will be stored in: {log_dir}")
    
    # Print configuration
    logger.info("Starting GPU Management API Server")
    logger.info(f"Configuration:")
    logger.info(f"  Host: {args.host}")
    logger.info(f"  Port: {args.port}")
    # logger.info(f"  Workers: {args.workers}")
    logger.info(f"  GPU refresh interval: {args.refresh_interval}s")
    logger.info(f"  Max task wait time: {args.max_wait_time} minutes")
    logger.info(f"  Log level: {args.log_level}")
    logger.info(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    # Start server
    try:
        server = GPUAPIServer(
            host=args.host,
            port=args.port,
            log_dir=str(log_dir),
            logger=logger
        )
        
        logger.info("Starting server...")
        await server.start()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 