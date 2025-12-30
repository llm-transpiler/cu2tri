#!/usr/bin/env python3
"""
Unsloth RoPE Embedding 多形状测试脚本
使用与llm_trans兼容的输入格式
"""
import sys
import os
import argparse
import time
import torch
from pathlib import Path

# 添加当前目录到路径
TESTCASE_ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTCASE_ROOT_DIR))

# Add project root to path
PROJECT_ROOT = None
for candidate in TESTCASE_ROOT_DIR.resolve().parents:
    if (candidate / "cu2til").exists():
        PROJECT_ROOT = candidate
        break
if PROJECT_ROOT and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from get_data import Params, get_all_cuda_torch_inputs, cuda_output_tensor_transform
from torch_.ref import torch_kernel
from triton_.kernel import triton_kernel

def parse_args():
    parser = argparse.ArgumentParser(description='Unsloth RoPE Embedding 多形状测试')
    parser.add_argument('--no-perf', action='store_true',
                       help='禁用性能测试 (默认: False)')
    parser.add_argument('--shapes', nargs='+', type=int,
                        help='自定义测试形状 (例如: --shapes 1 512 32 64)')
    parser.add_argument('--gpu-index', type=int, default=None,
                        help='强制设置GPU设备索引')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='数值比较容差')
    parser.add_argument('--warmup', type=int, default=10,
                        help='性能测试预热迭代次数')
    parser.add_argument('--iters', type=int, default=100,
                        help='性能测试迭代次数')
    parser.add_argument('--verbose', action='store_true',
                        help='启用详细输出')
    return parser.parse_args()

def test_unsloth_rope_multiple_shapes(params: Params, args):
    """测试Unsloth RoPE embedding kernel的多个形状"""
    all_test_data = get_all_cuda_torch_inputs(params)
    all_passed = True

    print(f"🧪 测试 {len(all_test_data)} 个形状...")
    print("=" * 80)

    for i, (shape, test_bundle) in enumerate(all_test_data):
        cuda_all_inputs, torch_all_inputs, cuda_output_tensors = test_bundle

        print(f"\n📐 测试形状 {i+1}/{len(all_test_data)}: {shape}")
        print("-" * 60)

        if args.verbose:
            for j, inp in enumerate(cuda_all_inputs):
                if isinstance(inp, torch.Tensor):
                    print(f"   输入 {j}: {inp.shape}, dtype: {inp.dtype}")

        try:
            # 运行Triton kernel
            print("🔥 运行Triton kernel...")
            start_time = time.perf_counter()

            triton_result = triton_kernel(*cuda_all_inputs)
            torch.cuda.synchronize()

            triton_time = (time.perf_counter() - start_time) * 1000

            # 运行PyTorch参考实现
            print("⚡ 运行PyTorch参考...")
            start_time = time.perf_counter()

            torch_result = torch_kernel(*torch_all_inputs)
            torch.cuda.synchronize()

            torch_time = (time.perf_counter() - start_time) * 1000

            # 比较结果
            diff = (triton_result - torch_result).abs()
            max_diff = diff.max().item()

            if max_diff > args.tolerance:
                print(f"❌ 数值不匹配! 最大差异: {max_diff:.6f} (容差: {args.tolerance})")
                if args.verbose:
                    print(f"   Triton范围: [{triton_result.min().item():.6f}, {triton_result.max().item():.6f}]")
                    print(f"   Torch范围: [{torch_result.min().item():.6f}, {torch_result.max().item():.6f}]")
                all_passed = False
            else:
                print(f"✅ 数值匹配! 最大差异: {max_diff:.6f}")

            # 性能测试
            if not args.no_perf:
                print(f"📊 性能对比:")
                print(f"   Triton: {triton_time:.3f} ms")
                print(f"   Torch:  {torch_time:.3f} ms")

                speedup = torch_time / triton_time
                print(f"   加速比: {speedup:.2f}x")

                # 计算带宽 (假设float32, 4 bytes per element)
                total_elements = torch.prod(torch.tensor(shape)).item()
                if total_elements > 0:
                    # 假设读写1个输入Q + 1个输出 = 2 * total_elements * 4 bytes
                    bandwidth = (2 * total_elements * 4) / (triton_time / 1000) / 1e9
                    print(f"   带宽: {bandwidth:.2f} GB/s")

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            all_passed = False

    return all_passed

def main():
    args = parse_args()

    # 设置GPU设备
    if args.gpu_index is not None:
        torch.cuda.set_device(args.gpu_index)
        print(f"🎮 使用GPU {args.gpu_index}: {torch.cuda.get_device_name()}")
    else:
        print(f"🎮 使用GPU {torch.cuda.current_device()}: {torch.cuda.get_device_name()}")

    # 创建参数对象
    params = Params()

    # 如果用户指定了自定义形状，使用它们
    if args.shapes:
        if len(args.shapes) == 4:
            # 4个数字，作为单个形状 (batch, seq_len, n_heads, head_dim)
            params.test_shapes = [tuple(args.shapes)]
        else:
            print("❌ RoPE embedding需要4个形状参数: batch seq_len n_heads head_dim")
            return False

        print(f"📋 使用自定义形状: {params.test_shapes}")
    else:
        print(f"📋 使用默认形状数量: {len(params.test_shapes)}")

    # 运行测试
    print("\n" + "=" * 80)
    print("🚀 Unsloth RoPE Embedding 多形状测试开始")
    print("=" * 80)

    success = test_unsloth_rope_multiple_shapes(params, args)

    print("\n" + "=" * 80)
    if success:
        print("🎉 所有测试通过!")
    else:
        print("💥 部分测试失败!")
    print("=" * 80)

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)