#!/usr/bin/env python3
"""
GPU Task Management Server Startup Script

This script starts the GPU Task Management API server with configurable options.
"""

import asyncio
import argparse
import logging
import signal
import sys
import os
from typing import List
# from api.server import GPUAPIServer
# from base.gpu_info import get_available_gpus
# Add the parent directory to Python path for proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import using absolute paths
from nvgpu_new.api.server import GPUAPIServer
from nvgpu_new.base.gpu_info import get_available_gpus


def parse_available_gpus(gpu_str: str) -> List[int]:
    """Parse comma-separated GPU IDs"""
    if not gpu_str:
        return []
    
    try:
        return [int(gpu_id.strip()) for gpu_id in gpu_str.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid GPU ID format: {e}")


def setup_logging(log_level: str) -> logging.Logger:
    """Setup logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('gpu_server.log')
        ]
    )
    
    # Reduce verbosity of some third-party libraries
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.error').setLevel(logging.INFO)
    
    return logging.getLogger(__name__)


async def main():
    """Main server startup function"""
    parser = argparse.ArgumentParser(
        description='GPU Task Management Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with default settings (use all GPUs)
  python start_server.py

  # Start server using only GPUs 0,1
  python start_server.py --available-gpus 0,1

  # Start server with custom settings
  python start_server.py --host 0.0.0.0 --port 8080 --max-parallel 8

  # Start server with custom CUDA error protection
  python start_server.py --error-cooldown 60
        """
    )
    
    # Server configuration
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to bind the server to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port to bind the server to (default: 8080)')
    
    # GPU configuration
    parser.add_argument('--available-gpus', type=parse_available_gpus, default=[],
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1"). Empty means use all GPUs')
    parser.add_argument('--max-parallel', type=int, default=4,
                        help='Maximum parallel tasks per GPU (default: 4)')
    
    # CUDA error protection
    parser.add_argument('--error-cooldown', type=int, default=30,
                        help='CUDA error cooldown period in seconds (default: 30)')
    
    # System configuration
    parser.add_argument('--refresh-interval', type=int, default=30,
                        help='GPU manager refresh interval in seconds (default: 30)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Display startup information
        logger.info("=" * 60)
        logger.info("[Server] GPU Task Management Server Starting")
        logger.info("=" * 60)
        
        # Show GPU configuration
        available_gpus = get_available_gpus(available_gpu_ids=args.available_gpus)
        if available_gpus:
            logger.info(f"[Server] Available GPUs: {list(available_gpus.keys())}")
            for gpu_id, gpu_info in available_gpus.items():
                logger.info(f"[Server]   GPU-{gpu_id}: {gpu_info.name} ({gpu_info.memory_total_mb}MB)")
        else:
            logger.warning("[Server] No GPUs available!")
        
        if args.available_gpus:
            logger.info(f"[Server] Using specific GPUs: {args.available_gpus}")
        else:
            logger.info("[Server] Using all available GPUs")
        
        # Show server configuration
        logger.info(f"[Server] Server: {args.host}:{args.port}")
        logger.info(f"[Server] Max parallel tasks per GPU: {args.max_parallel}")
        logger.info(f"[Server] CUDA error cooldown: {args.error_cooldown} seconds")
        logger.info(f"[Server] Refresh interval: {args.refresh_interval} seconds")
        logger.info(f"[Server] Log level: {args.log_level}")
        
        # Create and start the API server
        server = GPUAPIServer(
            host=args.host,
            port=args.port,
            available_gpu_ids=args.available_gpus,
            max_parallel_task_num=args.max_parallel,
            log_level=args.log_level,
            logger=logger
        )
        
        # Handle graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"[Server] Received signal {signum}, shutting down...")
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the server
        logger.info("=" * 60)
        logger.info(f"[Server] Server running at http://{args.host}:{args.port}")
        logger.info("[Server] Press Ctrl+C to stop the server")
        logger.info("=" * 60)
        
        await server.start()
        
    except KeyboardInterrupt:
        logger.info("[Server] Server shutdown requested by user")
    except Exception as e:
        logger.error(f"[Server] Failed to start server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main()) 