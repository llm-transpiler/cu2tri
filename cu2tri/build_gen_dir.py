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
OUTPUT_ROOT = WORKSPACE_ROOT / "cu2tri" / "outputs" / "kernelbench_c"
levels = ['level1', 'level2', 'level3']
name_set = {'level1': set(), 'level2': set(), 'level3': set()}
def process():    
    for level in ['level1', 'level2', 'level3']:
        level_dir = CUDA_REF_DIR / level
        if not level_dir.exists():
            continue
        for file in level_dir.glob("*.cu"):
            name_set[level].add(file.stem)
    for level in levels:
        level_dir = TORCH_REF_DIR / level
        if not level_dir.exists():
            continue
        for file in level_dir.glob("*.py"):
            if file.stem not in name_set[level]:
                raise ValueError(f"File {level + '_' + file.stem} not found in {level_dir}")
    for level in levels:
        name_set[level] = sorted(list(name_set[level]), key=lambda x: int(x.split('_')[0])) # type: ignore
    for level in levels:
        os.makedirs(OUTPUT_ROOT / level, exist_ok=True)
        for name in name_set[level]:
            os.makedirs(OUTPUT_ROOT / level / f"{name}", exist_ok=True)
            shutil.copy(CUDA_REF_DIR / level / f"{name}.cu", OUTPUT_ROOT / level / f"{name}" / f"cuda_ref.cu")
            shutil.copy(TORCH_REF_DIR / level / f"{name}.py", OUTPUT_ROOT / level / f"{name}" / f"torch_ref.py")

if __name__ == "__main__":
    process()
    