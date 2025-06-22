#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天服务示例启动脚本
基于精简后的 /llm 模块
"""
import asyncio
import logging
import os
from pathlib import Path

# 设置环境变量（如果需要）
import dotenv
dotenv.load_dotenv()

from .api_server import create_chat_api_server

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """主函数"""
    # 检查环境变量
    required_env_vars = ["GEMINI_API_KEY", "OPENROUTER_API_KEY"]
    optional_env_vars = ["OPENAI_API_KEY"]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set these variables in your .env file or environment")
        return
    
    # 显示可用的API密钥
    available_apis = []
    for var in required_env_vars + optional_env_vars:
        if os.getenv(var):
            available_apis.append(var.replace("_API_KEY", ""))
    
    logger.info(f"Available APIs: {available_apis}")
    
    # 创建并启动服务器
    server = create_chat_api_server(
        host="0.0.0.0",
        port=8001,
        logger=logger
    )
    
    logger.info("=" * 60)
    logger.info("🚀 Chat Service v2.0 Starting")
    logger.info("=" * 60)
    logger.info("📊 API Documentation: http://localhost:8000/docs")
    logger.info("🔄 ReDoc Documentation: http://localhost:8000/redoc")
    logger.info("🏥 Health Check: http://localhost:8000/health")
    logger.info("📈 Service Stats: http://localhost:8000/stats")
    logger.info("🤖 Supported Models: http://localhost:8000/models")
    logger.info("=" * 60)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("\n🛑 Received shutdown signal")
        await server.stop()
        logger.info("👋 Chat service stopped")
    except Exception as e:
        logger.error(f"💥 Server error: {e}")
        await server.stop()

if __name__ == "__main__":
    asyncio.run(main()) 