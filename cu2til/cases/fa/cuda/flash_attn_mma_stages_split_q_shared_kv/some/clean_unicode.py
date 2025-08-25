#!/usr/bin/env python3
"""
清理 .cu 文件中的不可见 Unicode 字符
"""

import re
import sys
import os
import unicodedata

def clean_invisible_unicode(text):
    """移除文本中的不可见 Unicode 字符"""
    
    # 定义需要移除的不可见字符
    invisible_chars = [
        '\u200b',  # 零宽空格 ZERO WIDTH SPACE
        '\u200c',  # 零宽非连接符 ZERO WIDTH NON-JOINER  
        '\u200d',  # 零宽连接符 ZERO WIDTH JOINER
        '\u2060',  # 字连接抑制符 WORD JOINER
        '\u2061',  # 函数应用 FUNCTION APPLICATION
        '\u2062',  # 不可见乘号 INVISIBLE TIMES
        '\u2063',  # 不可见分隔符 INVISIBLE SEPARATOR
        '\u2064',  # 不可见加号 INVISIBLE PLUS
        '\ufeff',  # 字节顺序标记 BYTE ORDER MARK / ZERO WIDTH NO-BREAK SPACE
        '\u00a0',  # 不间断空格 NON-BREAKING SPACE
        '\u1680',  # 欧甘空格 OGHAM SPACE MARK
        '\u180e',  # 蒙古文元音分隔符 MONGOLIAN VOWEL SEPARATOR
        '\u2000',  # EN QUAD
        '\u2001',  # EM QUAD
        '\u2002',  # EN SPACE
        '\u2003',  # EM SPACE
        '\u2004',  # THREE-PER-EM SPACE
        '\u2005',  # FOUR-PER-EM SPACE
        '\u2006',  # SIX-PER-EM SPACE
        '\u2007',  # FIGURE SPACE
        '\u2008',  # PUNCTUATION SPACE
        '\u2009',  # THIN SPACE
        '\u200a',  # HAIR SPACE
        '\u202f',  # NARROW NO-BREAK SPACE
        '\u205f',  # MEDIUM MATHEMATICAL SPACE
        '\u3000',  # 表意文字空格 IDEOGRAPHIC SPACE
    ]
    
    # 移除指定的不可见字符
    cleaned_text = text
    for char in invisible_chars:
        cleaned_text = cleaned_text.replace(char, '')
    
    # 移除其他控制字符（除了换行符、制表符、回车符）
    cleaned_text = ''.join(char for char in cleaned_text 
                          if unicodedata.category(char)[0] != 'C' 
                          or char in ['\n', '\t', '\r'])
    
    return cleaned_text

def detect_invisible_chars(text):
    """检测文本中的不可见字符"""
    invisible_found = []
    
    for i, char in enumerate(text):
        # 检查是否为控制字符或格式字符
        if unicodedata.category(char)[0] == 'C' and char not in ['\n', '\t', '\r']:
            invisible_found.append((i, char, hex(ord(char)), unicodedata.name(char, 'UNKNOWN')))
        elif unicodedata.category(char) == 'Zs' and char != ' ':
            invisible_found.append((i, char, hex(ord(char)), unicodedata.name(char, 'UNKNOWN')))
    
    return invisible_found

def clean_cu_file(input_file, output_file=None):
    """清理 .cu 文件中的不可见字符"""
    
    if output_file is None:
        output_file = input_file
    
    # 读取文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # 如果 UTF-8 读取失败，尝试其他编码
        print(f"UTF-8 读取失败，尝试 UTF-8-sig...")
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    
    # 检测不可见字符
    invisible_chars = detect_invisible_chars(content)
    
    if invisible_chars:
        print(f"发现 {len(invisible_chars)} 个不可见字符:")
        for pos, char, hex_code, name in invisible_chars[:10]:  # 只显示前10个
            print(f"  位置 {pos}: {repr(char)} ({hex_code}) - {name}")
        if len(invisible_chars) > 10:
            print(f"  ... 还有 {len(invisible_chars) - 10} 个")
    else:
        print("未发现不可见字符")
        return
    
    # 清理内容
    cleaned_content = clean_invisible_unicode(content)
    
    # 统计移除的字符数
    removed_count = len(content) - len(cleaned_content)
    
    # 写入清理后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print(f"清理完成:")
    print(f"  输入文件: {input_file}")
    print(f"  输出文件: {output_file}")
    print(f"  原始长度: {len(content)} 字符")
    print(f"  清理后长度: {len(cleaned_content)} 字符") 
    print(f"  移除字符数: {removed_count}")

if __name__ == "__main__":
    # 默认清理 region_split.cu
    input_file = "region_split.cu"
    
    # 如果命令行提供了参数，使用提供的文件
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    # 输出文件路径（可选）
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        sys.exit(1)
    
    clean_cu_file(input_file, output_file)
