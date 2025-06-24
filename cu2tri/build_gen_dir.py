#!/usr/bin/env python3
"""
构建所有 torch_functionals 和 KernelBench_c 测试用例的参考文件
为每个测试用例生成对应的 torch_ref.py 和 cuda_ref.cu 文件
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Dict, Tuple

# 定义基础路径
WORKSPACE_ROOT = Path("/workspace")
TESTS_ROOT = WORKSPACE_ROOT / "tests" / "_cuda" / "HPCTransCompile"
TORCH_REF_DIR = TESTS_ROOT / "EvalEngine" / "torch_functionals"
CUDA_REF_DIR = TESTS_ROOT / "KernelBench_c"
OUTPUT_ROOT = WORKSPACE_ROOT / "cu2tri" / "outputs" / "cu2tri" / "kernelbench_c"

# 定义源文件夹到目标文件夹的映射
# src_dirs = ['level1', 'level2', 'level3']
src2tgt_mapping = {
    'level1': '01_single_op',
    'level2': '02_fused_op', 
    'level3': '03_network'
}

name_set = {k: set() for k in src2tgt_mapping.keys()}

def process():    
    for src_dir in src2tgt_mapping.keys():
        src_path = CUDA_REF_DIR / src_dir
        if not src_path.exists():
            continue
        for file in src_path.glob("*.cu"):
            name_set[src_dir].add(file.stem)
    for src_dir in src2tgt_mapping.keys():
        src_path = TORCH_REF_DIR / src_dir
        if not src_path.exists():
            continue
        for file in src_path.glob("*.py"):
            if file.stem not in name_set[src_dir]:
                raise ValueError(f"File {src_dir + '_' + file.stem} not found in {src_path}")
    for src_dir in src2tgt_mapping.keys():
        name_set[src_dir] = sorted(list(name_set[src_dir]), key=lambda x: int(x.split('_')[0])) # type: ignore
    for src_dir in src2tgt_mapping.keys():
        # 使用映射后的目标目录名
        tgt_dir = src2tgt_mapping[src_dir]
        os.makedirs(OUTPUT_ROOT / tgt_dir, exist_ok=True)
        for name in name_set[src_dir]:
            os.makedirs(OUTPUT_ROOT / tgt_dir / f"{name}", exist_ok=True)
            shutil.copy(CUDA_REF_DIR / src_dir / f"{name}.cu", OUTPUT_ROOT / tgt_dir / f"{name}" / f"cuda_ref.cu")
            shutil.copy(TORCH_REF_DIR / src_dir / f"{name}.py", OUTPUT_ROOT / tgt_dir / f"{name}" / f"torch_ref.py")

if __name__ == "__main__":
    process()
    