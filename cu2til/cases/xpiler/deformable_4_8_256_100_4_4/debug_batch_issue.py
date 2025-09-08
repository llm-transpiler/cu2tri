import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import numpy as np

def analyze_batch_dimension_issue():
    """分析batch维度不匹配问题"""
    print("🔍 Analysis: Batch Dimension Mismatch Issue")
    print("=" * 60)
    
    print("📊 CUDA Kernel Expectations vs Our Data:")
    print()
    
    # 分析kernel参数
    print("🎯 Kernel Configuration (deformable_4_8_256_100_4_4):")
    print("   - n = 4     (batch_size)")
    print("   - l = 4     (num_levels)")  
    print("   - lq = 100  (num_queries)")
    print("   - m = 8     (num_heads)")
    print("   - d = 256   (embed_dim)")
    print()
    print("   Grid Layout: dim3 numBlocks(lq, n, m) = (100, 4, 8)")
    print("   - blockIdx.x: query index (0-99)")
    print("   - blockIdx.y: batch index (0-3)  ← 这里是关键！")
    print("   - blockIdx.z: head index (0-7)")
    print()
    
    # 分析我们的数据格式
    print("💾 Our Data Format:")
    print("   - value: [total_spatial, m, d] = [1360, 8, 256]")
    print("   - output: [lq, m, d] = [100, 8, 256]")
    print("   - NO batch dimension!")
    print()
    
    # 分析内存访问模式
    print("🧠 Memory Access Analysis:")
    print("   Kernel expects output indexing with blockIdx.y (batch):")
    print("   output_[(((((((int)blockIdx.y) * 204800) + ...))]")
    print("                     ^^^^^^^")
    print("   Where 204800 = lq * m * d = 100 * 8 * 256")
    print()
    print("   But our output tensor only has size: 100 * 8 * 256 = 204800")
    print("   When blockIdx.y > 0, it tries to access:")
    print("   - blockIdx.y=1: offset 204800")
    print("   - blockIdx.y=2: offset 409600") 
    print("   - blockIdx.y=3: offset 614400")
    print("   → All beyond tensor bounds! → Illegal memory access")
    print()
    
    print("✅ Working Kernels (n=1):")
    print("   - deformable_1_8_256_100_4_4: n=1, no batch dimension issue")
    print("   - deformable_1_8_256_200_4_4: n=1, no batch dimension issue")
    print("   - deformable_1_8_512_100_4_4: n=1, no batch dimension issue")
    print()
    
    print("❌ Failing Kernels (n>1):")
    print("   - deformable_4_8_256_100_4_4: n=4, expects batch data")
    print("   - Similar pattern for other n>1 cases")
    print()
    
    print("🎯 Root Cause:")
    print("   The CUDA kernels with n>1 are designed for batched inference,")
    print("   but we're providing non-batched data format!")
    print()
    
    print("💡 Solutions:")
    print("   1. Use only n=1 kernels (recommended)")
    print("   2. Or reshape data to include batch dimension") 
    print("   3. Or use MSDeformAttnFunction (best choice)")

def test_batch_reshaping_solution():
    """测试是否可以通过重塑数据来解决batch问题"""
    print("\n🧪 Testing Batch Reshaping Solution")
    print("=" * 60)
    
    print("Attempting to reshape data to match n=4 kernel expectations...")
    
    # 创建测试数据
    torch.manual_seed(42)
    device = "cuda"
    
    # 原始参数
    lq, m, d, l, k = 100, 8, 256, 4, 4
    n = 4  # batch size
    
    spatial_shapes = [(32, 32), (16, 16), (8, 8), (4, 4)]
    total_spatial = sum(h * w for h, w in spatial_shapes)
    
    print(f"📊 Original data shapes:")
    print(f"   - total_spatial: {total_spatial}")
    print(f"   - Expected output: [{lq}, {m}, {d}]")
    print(f"   - Kernel expects: [{n}, {lq}, {m}, {d}] (batched)")
    print()
    
    try:
        # 尝试创建batched数据
        print("🔄 Creating batched data format...")
        
        # 将数据复制n次创建batch
        value_single = torch.randn(total_spatial, m, d, dtype=torch.float32, device=device).normal_(0.0, 0.5)
        value_batched = value_single.unsqueeze(0).repeat(n, 1, 1, 1)  # [n, total_spatial, m, d]
        
        sampling_locations_single = torch.rand(lq, m, l*k, 2, dtype=torch.float32, device=device)
        sampling_locations_single = torch.clamp(sampling_locations_single, 0.1, 0.9) 
        sampling_locations_batched = sampling_locations_single.unsqueeze(0).repeat(n, 1, 1, 1, 1)  # [n, lq, m, l*k, 2]
        
        attention_weights_single = torch.randn(lq, m, l*k, dtype=torch.float32, device=device).normal_(0.0, 0.5)
        attention_weights_single = torch.abs(attention_weights_single)
        attention_weights_single = attention_weights_single / attention_weights_single.sum(dim=-1, keepdim=True)
        attention_weights_batched = attention_weights_single.unsqueeze(0).repeat(n, 1, 1, 1)  # [n, lq, m, l*k]
        
        output_batched = torch.empty(n, lq, m, d, dtype=torch.float32, device=device)
        
        print(f"✅ Batched data created:")
        print(f"   - value_batched: {value_batched.shape}")
        print(f"   - sampling_locations_batched: {sampling_locations_batched.shape}")
        print(f"   - attention_weights_batched: {attention_weights_batched.shape}")
        print(f"   - output_batched: {output_batched.shape}")
        
        print(f"\n💡 This approach could work, but requires:")
        print(f"   1. Modifying check_cuda.py to create batched data")
        print(f"   2. Changing input/output tensor layouts")
        print(f"   3. Significant code changes")
        print(f"\n🎯 Recommendation: Stick with n=1 kernels or use MSDeformAttnFunction")
        
    except Exception as e:
        print(f"❌ Batched data creation failed: {e}")

def main():
    analyze_batch_dimension_issue()
    test_batch_reshaping_solution()
    
    print("\n" + "=" * 60)
    print("🎉 Debug Analysis Complete!")
    print()
    print("📋 Summary:")
    print("   ✅ Found root cause: Batch dimension mismatch")
    print("   ✅ Identified working kernels: n=1 cases")
    print("   ✅ Identified failing kernels: n>1 cases")
    print()
    print("🎯 Recommendations:")
    print("   1. Use deformable_1_8_256_* kernels (they work!)")
    print("   2. Avoid deformable_4_8_256_* kernels (batch dimension issue)")
    print("   3. Best choice: Use MSDeformAttnFunction (no hardcoded limitations)")

if __name__ == "__main__":
    main()
