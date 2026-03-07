#!/usr/bin/env python3
"""
自动构建 triton_tutorial testset 的脚本
从 Triton 官方 tutorials 提取算子并创建测试用例
"""

import os
import sys
import shutil
from pathlib import Path
import re

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
TUTORIALS_DIR = PROJECT_ROOT / "triton_tutorials" / "v340"
TARGET_DIR = PROJECT_ROOT / "llm_trans" / "cases" / "triton_tutorial"
TOOLS_DIR = PROJECT_ROOT / "llm_trans" / "tools"

# Tutorial 到测试用例的映射
TUTORIAL_MAPPINGS = [
    {
        'file': '01-vector-add.py',
        'case_name': 'vector_add',
        'kernel_func': 'add_kernel',
        'wrapper_func': 'add',
        'test_configs': [
            {'size': 1024, 'dtype': 'torch.float32'},
            {'size': 98432, 'dtype': 'torch.float32'},
            {'size': 1048576, 'dtype': 'torch.float32'},
        ],
        'torch_ref': 'lambda x, y: x + y',
    },
    {
        'file': '02-fused-softmax.py',
        'case_name': 'fused_softmax',
        'kernel_func': 'softmax_kernel',
        'wrapper_func': 'softmax',
        'test_configs': [
            {'n_rows': 1024, 'n_cols': 1024, 'dtype': 'torch.float32'},
            {'n_rows': 4096, 'n_cols': 4096, 'dtype': 'torch.float16'},
        ],
        'torch_ref': 'lambda x: torch.softmax(x, dim=-1)',
    },
    {
        'file': '05-layer-norm.py',
        'case_name': 'layer_norm',
        'kernel_func': '_layer_norm_fwd_fused',
        'wrapper_func': 'layer_norm',
        'test_configs': [
            {'M': 1024, 'N': 1024, 'dtype': 'torch.float32'},
            {'M': 4096, 'N': 4096, 'dtype': 'torch.float16'},
        ],
        'torch_ref': 'lambda x, weight, bias, eps: torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps)',
    },
]

def extract_kernel_from_tutorial(tutorial_file, kernel_func, wrapper_func):
    """从 tutorial 文件提取 kernel 代码"""
    with open(tutorial_file, 'r') as f:
        content = f.read()
    
    # 提取 kernel 函数
    kernel_pattern = rf'@triton\.jit\s+def {kernel_func}\([^)]+\):.*?(?=\n@|\ndef [a-z_]+\(|\nclass |\n# %%|\Z)'
    kernel_match = re.search(kernel_pattern, content, re.DOTALL)
    
    if not kernel_match:
        print(f"Warning: Could not find kernel {kernel_func} in {tutorial_file}")
        return None, None
    
    kernel_code = kernel_match.group(0)
    
    # 提取 wrapper 函数
    wrapper_pattern = rf'def {wrapper_func}\([^)]+\):.*?(?=\n\ndef [a-z_]+\(|\n@|\nclass |\n# %%|\Z)'
    wrapper_match = re.search(wrapper_pattern, content, re.DOTALL)
    
    wrapper_code = wrapper_match.group(0) if wrapper_match else None
    
    return kernel_code, wrapper_code


def create_test_case(mapping):
    """创建单个测试用例"""
    case_name = mapping['case_name']
    tutorial_file = TUTORIALS_DIR / mapping['file']
    
    if not tutorial_file.exists():
        print(f"❌ Tutorial file not found: {tutorial_file}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Creating test case: {case_name}")
    print(f"{'='*60}")
    
    # 创建目录结构
    case_dir = TARGET_DIR / case_name
    for subdir in ['triton_', 'torch_', 'cute_', 'logs']:
        (case_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # 提取 kernel 代码
    kernel_code, wrapper_code = extract_kernel_from_tutorial(
        tutorial_file, mapping['kernel_func'], mapping['wrapper_func']
    )
    
    if kernel_code is None:
        print(f"❌ Failed to extract kernel from {tutorial_file}")
        return False
    
    # 生成 triton_/kernel.py
    triton_code = f'''"""
{case_name} - Triton Implementation
Extracted from Triton Tutorial: {mapping['file']}
"""
import torch
import triton
import triton.language as tl


{kernel_code}


{wrapper_code if wrapper_code else f"# Wrapper function for {mapping['wrapper_func']} not extracted"}


def triton_kernel(*args, **kwargs):
    """Wrapper function for test harness."""
    return {mapping['wrapper_func']}(*args, **kwargs)
'''
    
    (case_dir / 'triton_' / 'kernel.py').write_text(triton_code)
    print(f"✅ Created triton_/kernel.py")
    
    # 生成 torch_/ref.py
    torch_code = f'''import torch
import torch.nn.functional as F

def torch_kernel(*args, **kwargs):
    """PyTorch reference implementation."""
    # {mapping['torch_ref']}
    {mapping['torch_ref']}
    return ({mapping['torch_ref']})(*args, **kwargs)
'''
    
    (case_dir / 'torch_' / 'ref.py').write_text(torch_code)
    print(f"✅ Created torch_/ref.py")
    
    # 生成 get_data.py
    test_configs_str = '[\n    ' + ',\n    '.join([str(cfg) for cfg in mapping['test_configs']]) + ',\n]'
    
    getdata_code = f'''import torch
import numpy as np

class Params:
    def __init__(self, **kwargs):
        # Set default values and override with kwargs
        defaults = {mapping['test_configs'][0]}
        defaults.update(kwargs)
        for k, v in defaults.items():
            if isinstance(v, str) and v.startswith('torch.'):
                v = eval(v)
            setattr(self, k, v)

# Multiple test configurations
TEST_CONFIGS = {test_configs_str}

def get_triton_torch_inputs(params: Params):
    """Generate test inputs."""
    torch.manual_seed(0)
    np.random.seed(0)
    
    # TODO: Customize based on algorithm
    # This is a placeholder - adjust for your specific kernel
    inputs = {{}}
    
    # Add your input generation logic here
    
    return inputs

def triton_output_tensor_transform(output):
    return output

# Compatibility aliases
get_cuda_torch_inputs = get_triton_torch_inputs
cuda_output_tensor_transform = triton_output_tensor_transform
'''
    
    (case_dir / 'get_data.py').write_text(getdata_code)
    print(f"✅ Created get_data.py (needs manual customization)")
    
    # 复制 check_cute.py
    shutil.copy(TOOLS_DIR / 'check_cute.py', case_dir / 'check_cute.py')
    print(f"✅ Copied check_cute.py from tools")
    
    print(f"✅ Test case '{case_name}' created successfully")
    print(f"   Location: {case_dir}")
    print(f"   ⚠️  Note: get_data.py needs manual customization for input generation")
    
    return True


def main():
    """主函数"""
    print("="*70)
    print("Triton Tutorial Testset Builder")
    print("="*70)
    print(f"Source: {TUTORIALS_DIR}")
    print(f"Target: {TARGET_DIR}")
    print("="*70)
    
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for mapping in TUTORIAL_MAPPINGS:
        if create_test_case(mapping):
            success_count += 1
    
    print(f"\n{'='*70}")
    print(f"Summary: Created {success_count}/{len(TUTORIAL_MAPPINGS)} test cases")
    print(f"{'='*70}")
    
    if success_count < len(TUTORIAL_MAPPINGS):
        print("\n⚠️  Some test cases failed. Please review the output above.")
        return 1
    
    print("\n✅ All test cases created successfully!")
    print("\n📝 Next steps:")
    print("1. Customize get_data.py for each case (input generation logic)")
    print("2. Verify each Triton kernel runs correctly")
    print("3. Update llm_trans/config/case_config.yaml")
    print("4. Run LLM translation tests")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

