#!/usr/bin/env python3
"""
GELU实现对比分析
比较tanh近似版本和erf精确版本的差异
"""

import torch
import numpy as np

def detailed_error_analysis(output_torch, output_cuda, input_x, shape):
    """详细分析两个tensor之间的误差"""
    print(f"\n🔍 详细误差分析:")
    print("=" * 50)
    
    # 计算误差
    abs_diff = torch.abs(output_torch - output_cuda)
    rel_diff = torch.abs((output_torch - output_cuda) / (torch.abs(output_torch) + 1e-8))
    
    # 基本统计
    print(f"📊 基本统计:")
    print(f"   绝对误差 - 最大值: {abs_diff.max().item():.6e}")
    print(f"   绝对误差 - 平均值: {abs_diff.mean().item():.6e}")
    print(f"   绝对误差 - 中位数: {abs_diff.median().item():.6e}")
    print(f"   相对误差 - 最大值: {rel_diff.max().item():.6e}")
    print(f"   相对误差 - 平均值: {rel_diff.mean().item():.6e}")
    print(f"   相对误差 - 中位数: {rel_diff.median().item():.6e}")
    
    # 找到最大绝对误差的位置
    max_abs_idx = torch.argmax(abs_diff)
    max_abs_pos = np.unravel_index(max_abs_idx.cpu().numpy(), shape)
    max_abs_error = abs_diff.flatten()[max_abs_idx].item()
    
    print(f"\n🎯 最大绝对误差详情:")
    print(f"   位置: {max_abs_pos}")
    print(f"   扁平化索引: {max_abs_idx.item()}")
    print(f"   输入值: {input_x[max_abs_pos].item():.6f}")
    print(f"   PyTorch输出: {output_torch[max_abs_pos].item():.6f}")
    print(f"   CUDA输出: {output_cuda[max_abs_pos].item():.6f}")
    print(f"   绝对误差: {max_abs_error:.6e}")
    print(f"   相对误差: {rel_diff.flatten()[max_abs_idx].item():.6e}")
    
    # 找到最大相对误差的位置
    max_rel_idx = torch.argmax(rel_diff)
    max_rel_pos = np.unravel_index(max_rel_idx.cpu().numpy(), shape)
    max_rel_error = rel_diff.flatten()[max_rel_idx].item()
    
    print(f"\n🎯 最大相对误差详情:")
    print(f"   位置: {max_rel_pos}")
    print(f"   扁平化索引: {max_rel_idx.item()}")
    print(f"   输入值: {input_x[max_rel_pos].item():.6f}")
    print(f"   PyTorch输出: {output_torch[max_rel_pos].item():.6f}")
    print(f"   CUDA输出: {output_cuda[max_rel_pos].item():.6f}")
    print(f"   绝对误差: {abs_diff.flatten()[max_rel_idx].item():.6e}")
    print(f"   相对误差: {max_rel_error:.6e}")
    
    # 误差分布分析
    print(f"\n📈 误差分布分析:")
    abs_thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for thresh in abs_thresholds:
        count = (abs_diff > thresh).sum().item()
        percentage = count / abs_diff.numel() * 100
        print(f"   绝对误差 > {thresh:.0e}: {count:5d} 个 ({percentage:5.2f}%)")
    
    rel_thresholds = [1e-4, 1e-3, 1e-2, 1e-1, 0.5]
    for thresh in rel_thresholds:
        count = (rel_diff > thresh).sum().item()
        percentage = count / rel_diff.numel() * 100
        print(f"   相对误差 > {thresh:.0e}: {count:5d} 个 ({percentage:5.2f}%)")
    
    # 输出前10个最大误差的详情
    print(f"\n📋 前10个最大绝对误差:")
    _, top_indices = torch.topk(abs_diff.flatten(), k=min(10, abs_diff.numel()))
    for i, idx in enumerate(top_indices):
        pos = np.unravel_index(idx.cpu().numpy(), shape)
        print(f"   {i+1:2d}. 位置{pos}, 输入:{input_x[pos].item():7.4f}, "
              f"PyTorch:{output_torch[pos].item():8.5f}, "
              f"CUDA:{output_cuda[pos].item():8.5f}, "
              f"误差:{abs_diff.flatten()[idx].item():.6e}")
    
    # 判断结果是否匹配
    max_allowed_error = 1e-4
    if abs_diff.max().item() <= max_allowed_error:
        print(f"\n✅ 结果匹配 (最大误差 {abs_diff.max().item():.6e} <= {max_allowed_error:.0e})")
    else:
        print(f"\n❌ 结果不匹配 (最大误差 {abs_diff.max().item():.6e} > {max_allowed_error:.0e})")

def gelu_tanh_approx(x):
    """GELU的tanh近似实现（CUDA kernel中使用的版本）"""
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / np.pi)) * (x + 0.044715 * torch.pow(x, 3))))

def gelu_erf_exact(x):
    """GELU的erf精确实现（PyTorch默认版本）"""
    return torch.nn.functional.gelu(x)

def analyze_gelu_implementations():
    """分析两种GELU实现的差异"""
    print("🔍 GELU实现对比分析")
    print("=" * 60)
    
    # 测试不同范围的输入值
    test_ranges = [
        ("小正值", torch.linspace(0, 1, 100)),
        ("大正值", torch.linspace(1, 4, 100)),
        ("小负值", torch.linspace(-1, 0, 100)),
        ("大负值", torch.linspace(-4, -1, 100)),
        ("问题区间", torch.linspace(-4, 4, 1000))
    ]
    
    for name, x_values in test_ranges:
        print(f"\n📊 {name} 分析:")
        
        # 计算两种实现的结果
        y_tanh = gelu_tanh_approx(x_values)
        y_erf = gelu_erf_exact(x_values)
        
        # 计算误差
        abs_diff = torch.abs(y_tanh - y_erf)
        rel_diff = torch.abs((y_tanh - y_erf) / (torch.abs(y_erf) + 1e-8))
        
        # 统计信息
        print(f"   最大绝对误差: {abs_diff.max().item():.6e}")
        print(f"   平均绝对误差: {abs_diff.mean().item():.6e}")
        print(f"   最大相对误差: {rel_diff.max().item():.6e}")
        print(f"   平均相对误差: {rel_diff.mean().item():.6e}")
        
        # 找到最大误差的位置
        max_abs_idx = torch.argmax(abs_diff)
        max_rel_idx = torch.argmax(rel_diff)
        
        print(f"   最大绝对误差位置: x={x_values[max_abs_idx].item():.4f}")
        print(f"     tanh: {y_tanh[max_abs_idx].item():.6f}")
        print(f"     erf:  {y_erf[max_abs_idx].item():.6f}")
        
        if max_abs_idx != max_rel_idx:
            print(f"   最大相对误差位置: x={x_values[max_rel_idx].item():.4f}")
            print(f"     tanh: {y_tanh[max_rel_idx].item():.6f}")
            print(f"     erf:  {y_erf[max_rel_idx].item():.6f}")

def analyze_problematic_values():
    """分析实际测试中出现的问题值"""
    print(f"\n🎯 问题值专项分析:")
    print("=" * 60)
    
    # 从测试结果中提取的问题值
    problematic_inputs = torch.tensor([
        2.716126,   # 最大绝对误差位置的输入
        -3.686191,  # 最大相对误差位置的输入
        -2.7239,    # 其他高误差位置
        -2.6707,
        2.7335,
        -2.6584
    ])
    
    print(f"分析 {len(problematic_inputs)} 个问题输入值:")
    
    for i, x in enumerate(problematic_inputs):
        tanh_result = gelu_tanh_approx(x)
        erf_result = gelu_erf_exact(x)
        
        abs_error = torch.abs(tanh_result - erf_result).item()
        rel_error = torch.abs((tanh_result - erf_result) / (torch.abs(erf_result) + 1e-8)).item()
        
        print(f"\n   {i+1}. 输入值: {x.item():8.6f}")
        print(f"      tanh实现: {tanh_result.item():10.6f}")
        print(f"      erf实现:  {erf_result.item():10.6f}")
        print(f"      绝对误差: {abs_error:.6e}")
        print(f"      相对误差: {rel_error:.6e}")

def recommend_solutions():
    """推荐解决方案"""
    print(f"\n💡 解决方案建议:")
    print("=" * 60)
    print("1. 🎯 精度优先方案：")
    print("   - 将CUDA kernel改为使用erf实现")
    print("   - 可以获得与PyTorch完全一致的结果")
    print("   - 代码示例: 0.5 * x * (1.0 + erf(x * 0.7071067811865476))")
    print()
    print("2. ⚡ 性能优先方案：")
    print("   - 继续使用tanh近似，但调整测试容差")
    print("   - 将容差从1e-4调整到5e-4")
    print("   - tanh近似通常更快")
    print()
    print("3. 🔄 混合方案：")
    print("   - 对于绝对值较大的输入使用erf")
    print("   - 对于绝对值较小的输入使用tanh近似")
    print("   - 平衡精度和性能")

if __name__ == "__main__":
    analyze_gelu_implementations()
    analyze_problematic_values() 
    recommend_solutions()
