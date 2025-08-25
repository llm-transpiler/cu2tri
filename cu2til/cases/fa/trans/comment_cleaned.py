#!/usr/bin/env python3
"""
CUDA源文件预处理脚本
功能：
1. 删除所有注释（// 和 /* */ 类型）
2. 使用clang-format进行代码格式化
3. 生成 folder/kernel.cu -> folder/kernel_cleaned.cu 模式的输出文件
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def remove_comments(content):
    """
    删除C/C++/CUDA代码中的注释，保留字符串中的注释符号
    返回元组：(处理后的内容, 注释行索引集合)
    """
    result = []
    comment_line_indices = set()
    i = 0
    in_string = False
    in_char = False
    string_delimiter = None
    current_line = 0
    line_start = 0
    
    while i < len(content):
        char = content[i]
        
        # 处理字符串和字符字面量
        if not in_string and not in_char:
            if char == '"':
                in_string = True
                string_delimiter = '"'
                result.append(char)
            elif char == "'":
                in_char = True
                result.append(char)
            elif char == '/' and i + 1 < len(content):
                next_char = content[i + 1]
                if next_char == '/':
                    # 单行注释，跳过到行末
                    line_content = content[line_start:i].strip()
                    if line_content == '':
                        comment_line_indices.add(current_line)
                    
                    while i < len(content) and content[i] != '\n':
                        i += 1
                    continue
                elif next_char == '*':
                    # 多行注释，跳过到 */
                    line_content = content[line_start:i].strip()
                    if line_content == '':
                        comment_line_indices.add(current_line)
                    
                    i += 2
                    while i + 1 < len(content):
                        if content[i] == '*' and content[i + 1] == '/':
                            i += 2
                            break
                        i += 1
                    continue
                else:
                    result.append(char)
            else:
                result.append(char)
        else:
            # 在字符串或字符字面量中
            if in_string:
                result.append(char)
                if char == string_delimiter and (i == 0 or content[i-1] != '\\'):
                    # 检查是否是转义的引号
                    escape_count = 0
                    j = i - 1
                    while j >= 0 and content[j] == '\\':
                        escape_count += 1
                        j -= 1
                    if escape_count % 2 == 0:  # 偶数个反斜杠，引号未被转义
                        in_string = False
                        string_delimiter = None
            elif in_char:
                result.append(char)
                if char == "'" and (i == 0 or content[i-1] != '\\'):
                    # 检查是否是转义的单引号
                    escape_count = 0
                    j = i - 1
                    while j >= 0 and content[j] == '\\':
                        escape_count += 1
                        j -= 1
                    if escape_count % 2 == 0:  # 偶数个反斜杠，引号未被转义
                        in_char = False
        
        # 更新行号
        if char == '\n':
            current_line += 1
            line_start = i + 1
        
        i += 1
    
    return ''.join(result), comment_line_indices


def clean_empty_lines(content, comment_line_indices):
    """
    清理多余的空行，只删除原本是注释的行（除了注释符号外都是空白字符）
    """
    lines = content.split('\n')
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        # 只删除原本是注释的行
        if i in comment_line_indices:
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def remove_trailing_whitespace(content):
    """
    删除每行最后一个非空白字符之后的空白字符（尾随空白字符）
    """
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # 删除行尾空白字符
        cleaned_line = line.rstrip()
        cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)


def format_with_clang_format(file_path, style='Google'):
    """
    使用clang-format对文件进行格式化
    """
    try:
        # 检查clang-format是否可用
        subprocess.run(['clang-format', '--version'], capture_output=True, check=True)
        
        # 执行格式化
        result = subprocess.run([
            'clang-format', 
            f'-style={style}', 
            '-i',  # 就地修改
            str(file_path)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"警告: clang-format 格式化失败: {result.stderr}", file=sys.stderr)
            return False
        
        return True
    except subprocess.CalledProcessError:
        print("警告: clang-format 不可用，跳过代码格式化", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("警告: 未找到 clang-format，跳过代码格式化", file=sys.stderr)
        return False


def process_cuda_file(input_file, output_file=None, format_code=True, clang_format_style='Google'):
    """
    处理单个CUDA文件：删除注释 + 格式化
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"错误: 输入文件 {input_path} 不存在", file=sys.stderr)
        return False
    
    # 确定输出文件路径
    if output_file:
        output_path = Path(output_file)
    else:
        # 默认生成 kernel_cleaned.cu
        if input_path.name == 'kernel.cu':
            output_path = input_path.parent / 'kernel_cleaned.cu'
        else:
            # 其他文件名则添加 _cleaned 后缀
            output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
    
    try:
        print(f"处理文件: {input_path}")
        
        # 读取输入文件
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 删除注释
        print("  -> 删除注释...")
        content_without_comments, comment_line_indices = remove_comments(content)
        
        # 清理空行
        content_without_comments = clean_empty_lines(content_without_comments, comment_line_indices)
        
        # 删除尾随空白字符
        content_without_comments = remove_trailing_whitespace(content_without_comments)
        
        # 写入临时文件
        temp_output_path = output_path.with_suffix('.tmp' + output_path.suffix)
        with open(temp_output_path, 'w', encoding='utf-8') as f:
            f.write(content_without_comments)
        
        # 格式化代码
        if format_code:
            print(f"  -> 格式化代码 (样式: {clang_format_style})...")
            if format_with_clang_format(temp_output_path, clang_format_style):
                print("  -> 格式化成功")
            else:
                print("  -> 格式化失败，但文件仍已处理")
        
        # 移动到最终位置
        temp_output_path.rename(output_path)
        
        # 统计信息
        original_lines = len(content.split('\n'))
        with open(output_path, 'r', encoding='utf-8') as f:
            cleaned_content = f.read()
        cleaned_lines = len(cleaned_content.split('\n'))
        
        print(f"  -> 处理完成: {output_path}")
        print(f"     原文件行数: {original_lines}")
        print(f"     处理后行数: {cleaned_lines}")
        print(f"     减少行数: {original_lines - cleaned_lines}")
        print(f"     文件大小: {input_path.stat().st_size} -> {output_path.stat().st_size} bytes")
        
        return True
        
    except Exception as e:
        print(f"错误: 处理文件 {input_path} 时发生错误: {e}", file=sys.stderr)
        # 清理临时文件
        temp_output_path = output_path.with_suffix('.tmp' + output_path.suffix)
        if temp_output_path.exists():
            temp_output_path.unlink()
        return False


def process_folder(folder_path, format_code=True, clang_format_style='Google'):
    """
    处理文件夹中的kernel.cu文件，生成kernel_cleaned.cu
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"错误: 文件夹 {folder} 不存在", file=sys.stderr)
        return False
    
    if not folder.is_dir():
        print(f"错误: {folder} 不是一个文件夹", file=sys.stderr)
        return False
    
    kernel_file = folder / 'kernel.cu'
    
    if not kernel_file.exists():
        print(f"错误: 文件夹 {folder} 中没有找到 kernel.cu 文件", file=sys.stderr)
        return False
    
    output_file = folder / 'kernel_cleaned.cu'
    
    print(f"处理文件夹: {folder}")
    print(f"输入文件: {kernel_file}")
    print(f"输出文件: {output_file}")
    
    return process_cuda_file(
        input_file=str(kernel_file),
        output_file=str(output_file),
        format_code=format_code,
        clang_format_style=clang_format_style
    )


def main():
    parser = argparse.ArgumentParser(
        description='CUDA文件预处理：删除注释并格式化代码',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理文件夹中的kernel.cu（自动生成 kernel_cleaned.cu）
  python preprocess.py folder/
  
  # 处理单个文件并指定输出文件名
  python preprocess.py input.cu -o output_cleaned.cu
  
  # 处理文件夹但不格式化
  python preprocess.py folder/ --no-format
  
  # 使用不同的格式化样式
  python preprocess.py folder/ --format-style LLVM
        """
    )
    
    parser.add_argument('-i', '--input', 
                       help='输入路径：文件夹路径（处理其中的kernel.cu）或.cu文件路径')
    parser.add_argument('-o', '--output', 
                       help='输出文件路径（仅在输入为单个文件时有效）')
    parser.add_argument('--no-format', action='store_true',
                       help='跳过代码格式化步骤')
    parser.add_argument('--format-style', default='Google',
                       choices=['LLVM', 'Google', 'Chromium', 'Mozilla', 'WebKit'],
                       help='clang-format 格式化样式 (默认: Google)')
    
    args = parser.parse_args()
    input_path = args.input
    output = args.output
    if input_path is None:
        current_dir = Path(__file__).parent
        input_path = current_dir / '..' / 'cuda' / 'flash_attn_mma_stages_split_q_shared_kv' / 'ref.cu'
        output = current_dir / '..' / 'cuda' / 'flash_attn_mma_stages_split_q_shared_kv' / 'kernel.cu'
        # input_path = current_dir / "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages"
    input_path = Path(input_path)
    # 判断输入是文件夹还是文件
    if input_path.is_dir():
        # 处理文件夹中的kernel.cu
        success = process_folder(
            folder_path=input_path,
            format_code=not args.no_format,
            clang_format_style=args.format_style
        )
    elif input_path.is_file() or input_path.suffix == '.cu':
        # 处理单个文件
        success = process_cuda_file(
            input_file=input_path,
            output_file=output,
            format_code=not args.no_format,
            clang_format_style=args.format_style
        )
    else:
        print(f"错误: 输入路径 {input_path} 既不是文件夹也不是.cu文件", file=sys.stderr)
        success = False
    
    if success:
        print("\n✅ 预处理完成！")
        sys.exit(0)
    else:
        print("\n❌ 预处理失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()
