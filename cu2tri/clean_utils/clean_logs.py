#!/usr/bin/env python3
"""
清理 cu2tri/outputs/cu2tri/kernelbench_c/01_single_op 目录下所有的 logs 文件夹
"""

import os
import shutil
from pathlib import Path


def clean_logs_directories(base_path: str):
    """
    清理指定路径下所有子目录中的 logs 文件夹
    
    Args:
        base_path: 基础路径
    """
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        print(f"错误：路径 {base_path} 不存在")
        return
    
    # 统计变量
    total_found = 0
    total_removed = 0
    total_failed = 0
    
    print(f"开始清理 {base_path} 下的所有 logs 文件夹...")
    
    # 遍历所有子目录
    for sub_dir in base_dir.iterdir():
        if sub_dir.is_dir():
            logs_path = sub_dir / "logs"
            
            if logs_path.exists() and logs_path.is_dir():
                total_found += 1
                try:
                    # 计算文件夹大小（仅用于显示）
                    total_size = sum(f.stat().st_size for f in logs_path.glob('**/*') if f.is_file())
                    size_mb = total_size / (1024 * 1024)
                    
                    print(f"正在删除: {logs_path} (大小: {size_mb:.2f} MB)")
                    shutil.rmtree(logs_path)
                    total_removed += 1
                    print(f"✓ 已删除: {logs_path}")
                    
                except Exception as e:
                    total_failed += 1
                    print(f"✗ 删除失败: {logs_path} - 错误: {e}")
    
    # 输出统计信息
    print("\n" + "="*50)
    print("清理完成统计:")
    print(f"发现 logs 文件夹数量: {total_found}")
    print(f"成功删除数量: {total_removed}")
    print(f"删除失败数量: {total_failed}")
    print("="*50)


def main():
    """主函数"""
    # 默认路径
    default_path = "/workspace/cu2tri/outputs/cu2tri/kernelbench_c/01_single_op"
    
    # 也可以从命令行参数获取路径
    import sys
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        target_path = default_path
    
    # 确认操作
    print(f"即将清理路径: {target_path}")
    clean_logs_directories(target_path)


if __name__ == "__main__":
    main()
