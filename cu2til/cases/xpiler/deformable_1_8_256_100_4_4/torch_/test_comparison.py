import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn.functional as F
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

# Import the original and corrected implementations
from ms_deform_attn_func import ms_deform_attn_core_pytorch
from ref import torch_kernel, torch_kernel_vectorized

def create_test_data():
    """创建测试数据"""
    torch.manual_seed(42)
    
    # 参数设置
    lq = 100  # num_queries  
    m = 8     # num_heads
    d = 256   # embed_dim
    l = 4     # num_levels
    k = 4     # num_points per level
    
    # 定义4个level的空间形状 (height, width)
    spatial_shapes = [(32, 32), (16, 16), (8, 8), (4, 4)]
    value_spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.int32, device="cuda")
    
    # 计算每个level的起始索引
    level_starts = []
    total_spatial = 0
    for h, w in spatial_shapes:
        level_starts.append(total_spatial)
        total_spatial += h * w
    level_start_index = torch.tensor(level_starts, dtype=torch.int32, device="cuda")
    
    # 创建value tensor: [total_spatial_size, m, d]
    value = torch.randn(total_spatial, m, d, dtype=torch.float32, device="cuda") * 0.5
    
    # 创建采样位置: [lq, m, l*k, 2] (归一化坐标 0-1)
    sampling_locations = torch.rand(lq, m, l*k, 2, dtype=torch.float32, device="cuda")
    sampling_locations = torch.clamp(sampling_locations, 0.1, 0.9)
    
    # 创建注意力权重: [lq, m, l*k]  
    attention_weights = torch.randn(lq, m, l*k, dtype=torch.float32, device="cuda") * 0.5
    attention_weights = torch.abs(attention_weights)
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
    
    return (value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights)

def prepare_for_original_function(inputs):
    """为原始函数准备数据格式"""
    value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = inputs
    
    # 转换为原始函数期望的格式
    # value: [total_spatial, m, d] -> [N, S, M, D]
    N, S, M, D = 1, value.shape[0], value.shape[1], value.shape[2]
    value_orig = value.unsqueeze(0)  # [1, S, M, D]
    
    # sampling_locations: [lq, m, l*k, 2] -> [N, Lq, M, L, P, 2]
    lq, m, lk_2 = sampling_locations.shape
    l, k = 4, 4
    sampling_locs_orig = sampling_locations.view(lq, m, l, k, 2).unsqueeze(0)  # [1, Lq, M, L, P, 2]
    
    # attention_weights: [lq, m, l*k] -> [N, Lq, M, L, P]
    attn_weights_orig = attention_weights.view(lq, m, l, k).unsqueeze(0)  # [1, Lq, M, L, P]
    
    # 将spatial_shapes转换为list of tuples
    shapes_list = [(h.item(), w.item()) for h, w in value_spatial_shapes]
    
    return value_orig, shapes_list, sampling_locs_orig, attn_weights_orig

def test_implementations():
    """测试不同实现的结果"""
    print("🧪 测试不同的PyTorch实现...")
    print("=" * 60)
    
    # 创建测试数据
    inputs = create_test_data()
    value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = inputs
    
    print(f"📊 输入数据形状:")
    print(f"  value: {value.shape}")
    print(f"  value_spatial_shapes: {value_spatial_shapes.shape}")
    print(f"  level_start_index: {level_start_index.shape}")
    print(f"  sampling_locations: {sampling_locations.shape}")
    print(f"  attention_weights: {attention_weights.shape}")
    
    # 1. 测试我修正后的循环实现
    print(f"\n🔧 运行修正后的循环实现...")
    try:
        output_corrected = torch_kernel(*inputs)
        print(f"  输出形状: {output_corrected.shape}")
        print(f"  输出范围: [{output_corrected.min().item():.4f}, {output_corrected.max().item():.4f}]")
        print(f"  平均值: {output_corrected.mean().item():.4f}")
        print(f"  标准差: {output_corrected.std().item():.4f}")
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        output_corrected = None
    
    # 2. 测试向量化实现
    print(f"\n⚡ 运行向量化实现...")
    try:
        output_vectorized = torch_kernel_vectorized(*inputs)
        print(f"  输出形状: {output_vectorized.shape}")
        print(f"  输出范围: [{output_vectorized.min().item():.4f}, {output_vectorized.max().item():.4f}]")
        print(f"  平均值: {output_vectorized.mean().item():.4f}")
        print(f"  标准差: {output_vectorized.std().item():.4f}")
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        output_vectorized = None
    
    # 3. 测试原始的 ms_deform_attn_core_pytorch
    print(f"\n📚 运行原始 ms_deform_attn_core_pytorch...")
    try:
        value_orig, shapes_list, sampling_locs_orig, attn_weights_orig = prepare_for_original_function(inputs)
        output_original = ms_deform_attn_core_pytorch(value_orig, shapes_list, sampling_locs_orig, attn_weights_orig)
        # 调整输出形状 [1, Lq, M*D] -> [Lq, M, D]
        output_original = output_original.squeeze(0).view(100, 8, 256)
        print(f"  输出形状: {output_original.shape}")
        print(f"  输出范围: [{output_original.min().item():.4f}, {output_original.max().item():.4f}]")
        print(f"  平均值: {output_original.mean().item():.4f}")
        print(f"  标准差: {output_original.std().item():.4f}")
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        print(f"  原始函数需要的参数格式可能不匹配，跳过此测试")
        output_original = None
    
    # 4. 比较结果
    print(f"\n📊 结果比较:")
    if output_corrected is not None and output_vectorized is not None:
        diff_corrected_vectorized = (output_corrected - output_vectorized).abs()
        max_diff = diff_corrected_vectorized.max().item()
        mean_diff = diff_corrected_vectorized.mean().item()
        print(f"  循环实现 vs 向量化实现:")
        print(f"    最大差异: {max_diff:.2e}")
        print(f"    平均差异: {mean_diff:.2e}")
        print(f"    匹配度: {'✅ 高度一致' if max_diff < 1e-5 else '⚠️  存在差异' if max_diff < 1e-3 else '❌ 差异很大'}")
    
    if output_corrected is not None and output_original is not None:
        diff_corrected_original = (output_corrected - output_original).abs()
        max_diff = diff_corrected_original.max().item()
        mean_diff = diff_corrected_original.mean().item()
        print(f"  循环实现 vs 原始实现:")
        print(f"    最大差异: {max_diff:.2e}")
        print(f"    平均差异: {mean_diff:.2e}")
        print(f"    匹配度: {'✅ 高度一致' if max_diff < 1e-5 else '⚠️  存在差异' if max_diff < 1e-3 else '❌ 差异很大'}")
    
    if output_vectorized is not None and output_original is not None:
        diff_vectorized_original = (output_vectorized - output_original).abs()
        max_diff = diff_vectorized_original.max().item()
        mean_diff = diff_vectorized_original.mean().item()
        print(f"  向量化实现 vs 原始实现:")
        print(f"    最大差异: {max_diff:.2e}")
        print(f"    平均差异: {mean_diff:.2e}")
        print(f"    匹配度: {'✅ 高度一致' if max_diff < 1e-5 else '⚠️  存在差异' if max_diff < 1e-3 else '❌ 差异很大'}")
    
    # 5. 输出样本值进行详细比较
    print(f"\n🔬 样本值比较 (前3个元素):")
    if output_corrected is not None:
        print(f"  循环实现: {output_corrected.flatten()[:3]}")
    if output_vectorized is not None:
        print(f"  向量化实现: {output_vectorized.flatten()[:3]}")
    if output_original is not None:
        print(f"  原始实现: {output_original.flatten()[:3]}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，请确保有GPU环境")
    else:
        device = torch.cuda.current_device()
        print(f"🎮 使用GPU: {torch.cuda.get_device_name(device)}")
        test_implementations()
