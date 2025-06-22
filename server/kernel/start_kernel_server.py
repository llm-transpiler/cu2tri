#!/usr/bin/env python3
"""
内核开发服务器启动脚本
启动完整的内核开发服务器系统并进行集成检查
"""

import asyncio
import sys
import os
import signal
import logging
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from server.kernel.config.integration_config import (
    validate_config, create_directories, get_environment_info,
    get_config, LOGGING_CONFIG
)
from server.kernel.service import KernelDevelopmentService
from server.kernel.api_server import create_app
from server.kernel.gpu_manager import GPUManager
from server.kernel.queue_manager import QueueManager

# 全局变量
service: Optional[KernelDevelopmentService] = None
server_process = None

def setup_logging():
    """设置日志系统"""
    logging_config = get_config("logging")
    
    # 创建日志目录
    for log_file in logging_config["files"].values():
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置根日志器
    logging.basicConfig(
        level=getattr(logging, logging_config["level"]),
        format=logging_config["format"],
        handlers=[
            logging.FileHandler(logging_config["files"]["main"]),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("日志系统初始化完成")
    return logger

def signal_handler(signum, frame):
    """信号处理器"""
    logger = logging.getLogger(__name__)
    logger.info(f"接收到信号 {signum}，开始优雅关闭...")
    
    # 触发关闭事件
    asyncio.create_task(shutdown_server())

async def shutdown_server():
    """优雅关闭服务器"""
    logger = logging.getLogger(__name__)
    logger.info("开始关闭服务器...")
    
    global service, server_process
    
    try:
        # 关闭服务
        if service:
            await service.shutdown()
            logger.info("内核开发服务已关闭")
        
        # 关闭服务器进程
        if server_process:
            server_process.terminate()
            logger.info("API服务器已关闭")
        
    except Exception as e:
        logger.error(f"关闭服务器时出错: {e}")
    
    logger.info("服务器关闭完成")

async def check_system_requirements():
    """检查系统要求"""
    logger = logging.getLogger(__name__)
    logger.info("检查系统要求...")
    
    issues = []
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        issues.append(f"Python版本过低: {sys.version_info}, 需要3.8+")
    
    # 检查CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA不可用")
        else:
            gpu_count = torch.cuda.device_count()
            if gpu_count < 2:
                issues.append(f"GPU数量不足: {gpu_count}, 需要至少2个GPU")
            logger.info(f"检测到 {gpu_count} 个GPU")
    except ImportError:
        issues.append("PyTorch未安装")
    
    # 检查Triton
    try:
        import triton
        logger.info(f"Triton版本: {triton.__version__}")
    except ImportError:
        issues.append("Triton未安装")
    
    # 检查必要的包
    required_packages = [
        "fastapi", "uvicorn", "pydantic", "numpy", 
        "nvidia-ml-py3", "psutil", "requests"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"缺少包: {package}")
    
    if issues:
        logger.error("系统要求检查失败:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("✅ 系统要求检查通过")
    return True

async def initialize_components():
    """初始化系统组件"""
    logger = logging.getLogger(__name__)
    logger.info("初始化系统组件...")
    
    global service
    
    try:
        # 初始化GPU管理器
        logger.info("初始化GPU管理器...")
        gpu_manager = GPUManager()
        gpu_stats = await gpu_manager.get_gpu_stats()
        logger.info(f"GPU状态: {len(gpu_stats)} 个GPU可用")
        
        # 初始化队列管理器
        logger.info("初始化队列管理器...")
        queue_manager = QueueManager()
        
        # 初始化主服务
        logger.info("初始化内核开发服务...")
        service = KernelDevelopmentService()
        
        # 健康检查
        health = await service.health_check()
        if health["status"] != "healthy":
            logger.error(f"服务健康检查失败: {health}")
            return False
        
        logger.info("✅ 系统组件初始化完成")
        return True
        
    except Exception as e:
        logger.error(f"组件初始化失败: {e}")
        return False

async def run_integration_tests():
    """运行集成测试"""
    logger = logging.getLogger(__name__)
    logger.info("运行集成测试...")
    
    try:
        # 导入测试模块
        from server.kernel.tests.test_integration import main as integration_main
        
        # 运行集成测试
        test_result = await integration_main()
        
        if test_result == 0:
            logger.info("✅ 集成测试通过")
            return True
        else:
            logger.warning("⚠️  部分集成测试失败")
            return False
            
    except Exception as e:
        logger.error(f"集成测试异常: {e}")
        return False

def start_api_server():
    """启动API服务器"""
    logger = logging.getLogger(__name__)
    logger.info("启动API服务器...")
    
    try:
        import uvicorn
        from server.kernel.api_server import app
        
        api_config = get_config("api")
        
        # 在后台启动API服务器
        uvicorn.run(
            app,
            host=api_config["host"],
            port=api_config["port"],
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"API服务器启动失败: {e}")
        raise

async def monitor_system():
    """系统监控循环"""
    logger = logging.getLogger(__name__)
    monitoring_config = get_config("monitoring")
    
    while True:
        try:
            # 检查GPU状态
            if service:
                gpu_manager = service.gpu_manager
                gpu_stats = await gpu_manager.get_gpu_stats()
                
                # 检查温度警告
                for gpu_id, stats in gpu_stats.items():
                    temp = stats.get("temperature", 0)
                    if temp > monitoring_config["alerts"]["gpu_temperature_threshold"]:
                        logger.warning(f"GPU {gpu_id} 温度过高: {temp}°C")
                
                # 检查队列状态
                queue_manager = service.queue_manager
                queue_stats = queue_manager.get_queue_stats()
                
                pending_tasks = queue_stats.get("pending_tasks", 0)
                if pending_tasks > monitoring_config["alerts"]["queue_size_threshold"]:
                    logger.warning(f"队列任务过多: {pending_tasks}")
            
            # 等待下一次检查
            await asyncio.sleep(monitoring_config["health_check_interval"])
            
        except Exception as e:
            logger.error(f"系统监控异常: {e}")
            await asyncio.sleep(60)  # 出错时等待更长时间

async def main():
    """主函数"""
    logger = setup_logging()
    logger.info("🚀 启动内核开发服务器")
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 1. 验证配置
        logger.info("步骤 1/7: 验证配置...")
        if not validate_config():
            logger.error("配置验证失败")
            return 1
        
        # 2. 创建目录
        logger.info("步骤 2/7: 创建目录...")
        create_directories()
        
        # 3. 检查系统要求
        logger.info("步骤 3/7: 检查系统要求...")
        if not await check_system_requirements():
            logger.error("系统要求检查失败")
            return 1
        
        # 4. 初始化组件
        logger.info("步骤 4/7: 初始化组件...")
        if not await initialize_components():
            logger.error("组件初始化失败")
            return 1
        
        # 5. 运行集成测试（可选）
        if "--skip-tests" not in sys.argv:
            logger.info("步骤 5/7: 运行集成测试...")
            await run_integration_tests()
        else:
            logger.info("步骤 5/7: 跳过集成测试")
        
        # 6. 启动API服务器
        logger.info("步骤 6/7: 启动API服务器...")
        
        # 创建后台任务
        monitor_task = asyncio.create_task(monitor_system())
        
        try:
            # 启动API服务器（这会阻塞）
            start_api_server()
        except KeyboardInterrupt:
            logger.info("接收到中断信号")
        finally:
            # 取消监控任务
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("步骤 7/7: 服务器正常关闭")
        return 0
        
    except Exception as e:
        logger.error(f"启动失败: {e}")
        return 1
    finally:
        await shutdown_server()

def print_usage():
    """打印使用说明"""
    print("""
内核开发服务器启动脚本

用法:
    python start_kernel_server.py [选项]

选项:
    --skip-tests    跳过集成测试
    --help         显示此帮助信息

示例:
    python start_kernel_server.py
    python start_kernel_server.py --skip-tests

环境变量:
    OPENROUTER_API_KEY    OpenRouter API密钥
    GEMINI_API_KEY        Gemini API密钥
    KERNEL_ADMIN_KEY      管理员API密钥
    KERNEL_USER_KEY       用户API密钥

服务地址:
    API服务器: http://localhost:8000
    健康检查: http://localhost:8000/health
    API文档: http://localhost:8000/docs
    """)

if __name__ == "__main__":
    if "--help" in sys.argv:
        print_usage()
        sys.exit(0)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print(f"错误: Python版本 {sys.version_info} 太低，需要3.8或更高版本")
        sys.exit(1)
    
    # 运行主函数
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"致命错误: {e}")
        sys.exit(1) 