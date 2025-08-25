#!/usr/bin/env python3
"""
从 region_split_src.md 恢复 .cu 文件的简洁脚本
"""

import re
import sys
import os

def extract_cpp_blocks(md_content):
    """从 markdown 内容中提取所有 cpp 代码块"""
    # 匹配 ```cpp ... ``` 的代码块
    pattern = r'```cpp\n(.*?)\n```'
    matches = re.findall(pattern, md_content, re.DOTALL)
    return matches

def recover_cu_from_md(md_file_path, output_cu_path=None):
    """从 markdown 文件恢复 cu 文件"""
    # 如果没有指定输出路径，则使用默认名称
    if output_cu_path is None:
        base_name = os.path.splitext(md_file_path)[0]
        output_cu_path = base_name.replace('_src', '') + '.cu'
    
    # 读取 markdown 文件
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # 提取所有 cpp 代码块
    cpp_blocks = extract_cpp_blocks(md_content)
    
    if not cpp_blocks:
        print(f"警告: 在 {md_file_path} 中没有找到任何 cpp 代码块")
        return
    
    # 将所有代码块连接起来
    cu_content = '\n'.join(cpp_blocks)
    
    # 写入 .cu 文件
    with open(output_cu_path, 'w', encoding='utf-8') as f:
        f.write(cu_content)
    
    print(f"成功从 {md_file_path} 恢复到 {output_cu_path}")
    print(f"提取了 {len(cpp_blocks)} 个代码块")

if __name__ == "__main__":
    # 默认处理当前目录下的 region_split_src.md
    # md_file = "region_split_src.md"
    md_file = "region_split_tgt_prim.md"
    
    # 如果命令行提供了参数，使用提供的文件
    if len(sys.argv) > 1:
        md_file = sys.argv[1]
    
    # 输出文件路径（可选）
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(md_file):
        print(f"错误: 文件 {md_file} 不存在")
        sys.exit(1)
    
    recover_cu_from_md(md_file, output_file)