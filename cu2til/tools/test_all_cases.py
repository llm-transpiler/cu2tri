#!/usr/bin/env python3
"""
统一的多case测试脚本
支持FlagGems、Unsloth、Liger-Kernel和Triton Tutorials的所有测试用例
"""
import sys
import os
import argparse
import time
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
PROJECT_ROOT = None
for candidate in Path(__file__).resolve().parents:
    if (candidate / "cu2til").exists():
        PROJECT_ROOT = candidate
        break
if PROJECT_ROOT and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args():
    parser = argparse.ArgumentParser(description='统一的多case测试脚本')
    parser.add_argument('--testset', choices=['flaggems_ops', 'unsloth_kernels', 'ligerkernel_ops', 'triton_tutorials', 'all'],
                       default='all', help='选择测试集')
    parser.add_argument('--no-perf', action='store_true',
                       help='禁用性能测试')
    parser.add_argument('--gpu-index', type=int, default=None,
                       help='强制设置GPU设备索引')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                       help='数值比较容差')
    parser.add_argument('--warmup', type=int, default=5,
                       help='性能测试预热迭代次数')
    parser.add_argument('--iters', type=int, default=50,
                       help='性能测试迭代次数')
    parser.add_argument('--verbose', action='store_true',
                       help='启用详细输出')
    parser.add_argument('--use-nvgpu', action='store_true',
                       help='使用NVGPU server执行')
    parser.add_argument('--nvgpu-server', default='http://localhost:8080',
                       help='NVGPU server URL')
    return parser.parse_args()

def run_test_set(testset_name: str, test_root_dir: Path, args):
    """运行指定测试集的所有测试用例"""
    print(f"\n{'='*80}")
    print(f"🚀 测试集: {testset_name}")
    print(f"{'='*80}")

    all_passed = True
    total_cases = 0

    # 查找所有测试用例
    if test_root_dir.exists():
        test_cases = []
        for case_dir in test_root_dir.iterdir():
            if case_dir.is_dir() and (case_dir / "get_data.py").exists():
                test_cases.append(case_dir.name)

        test_cases.sort()
        print(f"📋 发现 {len(test_cases)} 个测试用例: {test_cases}")

        for case_name in test_cases:
            case_dir = test_root_dir / case_name

            # 添加当前case目录到path
            sys.path.insert(0, str(case_dir))

            try:
                print(f"\n🧪 测试用例: {case_name}")
                print("-" * 60)

                # 导入测试数据
                from get_data import Params, get_all_cuda_torch_inputs
                from torch_.ref import torch_kernel
                from triton_.kernel import triton_kernel

                params = Params()
                all_test_data = get_all_cuda_torch_inputs(params)

                case_passed = 0
                for i, (shape, test_bundle) in enumerate(all_test_data):
                    cuda_all_inputs, torch_all_inputs, cuda_output_tensors = test_bundle

                    if args.verbose:
                        print(f"   📐 形状 {i+1}/{len(all_test_data)}: {shape}")

                    try:
                        # 运行Triton kernel
                        start_time = time.perf_counter()
                        triton_result = triton_kernel(*cuda_all_inputs)
                        torch.cuda.synchronize()
                        triton_time = (time.perf_counter() - start_time) * 1000

                        # 运行PyTorch参考实现
                        start_time = time.perf_counter()
                        torch_result = torch_kernel(*torch_all_inputs)
                        torch.cuda.synchronize()
                        torch_time = (time.perf_counter() - start_time) * 1000

                        # 比较结果
                        diff = (triton_result - torch_result).abs()
                        max_diff = diff.max().item()

                        if max_diff > args.tolerance:
                            print(f"   ❌ 形状 {shape}: 最大差异 {max_diff:.6f} > {args.tolerance}")
                            all_passed = False
                        else:
                            case_passed += 1
                            if args.verbose:
                                print(f"   ✅ 形状 {shape}: 差异 {max_diff:.6f}")

                        # 性能统计
                        if not args.no_perf and case_passed > 0:
                            speedup = torch_time / triton_time if triton_time > 0 else 0
                            if args.verbose:
                                total_elements = torch.prod(torch.tensor(shape)).item()
                                bandwidth = (3 * total_elements * 4) / (triton_time / 1000) / 1e9 if total_elements > 0 else 0
                                print(f"      📊 加速比: {speedup:.2f}x, 带宽: {bandwidth:.2f} GB/s")

                    except Exception as e:
                        print(f"   ❌ 形状 {shape}: 错误 {e}")
                        all_passed = False

                total_cases += len(all_test_data)
                print(f"📊 通过 {case_passed}/{len(all_test_data)} 个形状")

                # 清理path
                if str(case_dir) in sys.path:
                    sys.path.remove(str(case_dir))

            except ImportError as e:
                print(f"❌ 导入失败: {e}")
                all_passed = False

    else:
        print(f"❌ 测试集目录不存在: {test_root_dir}")
        all_passed = False

    return all_passed, total_cases

def main():
    args = parse_args()

    # 设置GPU设备
    if args.gpu_index is not None:
        torch.cuda.set_device(args.gpu_index)
        print(f"🎮 使用GPU {args.gpu_index}: {torch.cuda.get_device_name()}")
    else:
        print(f"🎮 使用GPU {torch.cuda.current_device()}: {torch.cuda.get_device_name()}")

    # NVGPU配置
    if args.use_nvgpu:
        print(f"🌐 使用NVGPU server: {args.nvgpu_server}")
        # 这里可以添加NVGPU验证逻辑

    print("\n" + "="*80)
    print("🚀 统一多case测试开始")
    print("="*80)

    # 运行测试
    test_sets = []
    if args.testset == 'all':
        test_sets = ['flaggems_ops', 'unsloth_kernels', 'ligerkernel_ops', 'triton_tutorials']
    else:
        test_sets = [args.testset]

    total_passed = 0
    total_cases = 0

    for testset in test_sets:
        test_root_dir = PROJECT_ROOT / "cu2til/cases" / testset
        passed, cases = run_test_set(testset, test_root_dir, args)
        if passed:
            total_passed += 1
        total_cases += cases

    print(f"\n" + "="*80)
    print(f"📊 测试结果: {total_passed}/{len(test_sets)} 个测试集通过")
    print(f"   总计测试用例: {total_cases} 个形状")

    if total_passed == len(test_sets):
        print("🎉 所有测试通过!")
        return True
    else:
        print("💥 部分测试失败!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)