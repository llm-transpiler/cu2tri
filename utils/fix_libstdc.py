#!/usr/bin/env python3
#  conda install -c conda-forge libstdcxx-ng
"""
解决 libstdc++ 版本兼容性问题的脚本
"""
import os
import sys

def fix_libstdc_path():
    """
    修复 libstdc++ 路径问题
    让程序优先使用系统的 libstdc++ 而不是 conda 的旧版本
    """
    # 获取当前的 LD_LIBRARY_PATH
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    
    # 系统 libstdc++ 路径
    system_lib_paths = [
        '/usr/lib/x86_64-linux-gnu',
        '/lib/x86_64-linux-gnu',
        '/usr/lib64',
        '/lib64'
    ]
    
    # 将系统路径添加到 LD_LIBRARY_PATH 的前面
    new_paths = []
    for path in system_lib_paths:
        if os.path.exists(path):
            new_paths.append(path)
    
    if current_ld_path:
        new_ld_path = ':'.join(new_paths + [current_ld_path])
    else:
        new_ld_path = ':'.join(new_paths)
    
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    print(f"Updated LD_LIBRARY_PATH: {new_ld_path}")
    
    # 也可以尝试移除 conda 的 lib 路径
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        conda_lib = os.path.join(conda_prefix, 'lib')
        if conda_lib in new_ld_path:
            # 将 conda lib 移到最后
            paths = new_ld_path.split(':')
            paths = [p for p in paths if p != conda_lib] + [conda_lib]
            os.environ['LD_LIBRARY_PATH'] = ':'.join(paths)
            print(f"Moved conda lib to end: {os.environ['LD_LIBRARY_PATH']}")

if __name__ == "__main__":
    fix_libstdc_path()
    print("libstdc++ path fixed. You can now run your script.") 