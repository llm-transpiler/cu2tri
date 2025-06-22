#!/usr/bin/env python3
"""
测试运行脚本
运行内核开发服务器的各种测试
"""

import asyncio
import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def setup_environment():
    """设置测试环境"""
    # 添加路径
    os.environ['PYTHONPATH'] = str(project_root)
    
    # 设置GPU环境（如果需要）
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # 默认使用GPU 5

async def run_basic_tests():
    """运行基础测试"""
    print("🧪 运行基础测试...")
    try:
        from server.kernel.tests.test_basic import main
        result = await main()
        return result == 0
    except Exception as e:
        print(f"基础测试异常: {e}")
        return False

async def run_gpu5_tests():
    """运行GPU 5专项测试"""
    print("🧪 运行GPU 5专项测试...")
    try:
        from server.kernel.tests.test_gpu5 import main
        result = await main()
        return result == 0
    except Exception as e:
        print(f"GPU 5测试异常: {e}")
        return False

async def run_integration_tests():
    """运行集成测试"""
    print("🧪 运行集成测试...")
    try:
        from server.kernel.tests.test_integration import main
        result = await main()
        return result == 0
    except Exception as e:
        print(f"集成测试异常: {e}")
        return False

async def run_config_validation():
    """运行配置验证"""
    print("🧪 运行配置验证...")
    try:
        from server.kernel.config.integration_config import validate_config, get_environment_info
        
        # 验证配置
        config_valid = validate_config()
        
        # 显示环境信息
        env_info = get_environment_info()
        print("环境信息:")
        for key, value in env_info.items():
            print(f"  {key}: {value}")
        
        return config_valid
    except Exception as e:
        print(f"配置验证异常: {e}")
        return False

def print_test_summary(results):
    """打印测试摘要"""
    print("\n" + "="*60)
    print("测试摘要")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20} {status}")
    
    print("-"*60)
    print(f"总计: {passed_tests}/{total_tests} 通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过!")
    else:
        print("⚠️  部分测试失败")
    
    return passed_tests == total_tests

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="内核开发服务器测试运行器")
    parser.add_argument("--basic", action="store_true", help="只运行基础测试")
    parser.add_argument("--gpu5", action="store_true", help="只运行GPU 5测试")
    parser.add_argument("--integration", action="store_true", help="只运行集成测试")
    parser.add_argument("--config", action="store_true", help="只运行配置验证")
    parser.add_argument("--all", action="store_true", help="运行所有测试（默认）")
    parser.add_argument("--gpu-id", type=int, default=5, help="指定测试使用的GPU ID")
    
    args = parser.parse_args()
    
    # 设置GPU环境
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # 设置测试环境
    setup_environment()
    
    print(f"🚀 启动测试 (使用GPU {args.gpu_id})")
    print("="*60)
    
    results = []
    
    # 决定运行哪些测试
    if args.basic:
        tests_to_run = [("基础测试", run_basic_tests)]
    elif args.gpu5:
        tests_to_run = [("GPU 5测试", run_gpu5_tests)]
    elif args.integration:
        tests_to_run = [("集成测试", run_integration_tests)]
    elif args.config:
        tests_to_run = [("配置验证", run_config_validation)]
    else:
        # 默认运行所有测试
        tests_to_run = [
            ("配置验证", run_config_validation),
            ("基础测试", run_basic_tests),
            ("GPU 5测试", run_gpu5_tests),
            ("集成测试", run_integration_tests)
        ]
    
    # 运行测试
    for test_name, test_func in tests_to_run:
        print(f"\n--- {test_name} ---")
        try:
            passed = await test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"测试异常: {e}")
            results.append((test_name, False))
    
    # 打印摘要
    all_passed = print_test_summary(results)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"测试运行器异常: {e}")
        sys.exit(1) 