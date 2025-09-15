# 🚀 LeetCuda Dynamic Shape Testing Framework

## 概述

这是一个支持动态形状(Dynamic Shape)的CUDA kernel测试框架，可以使用多种不同的张量尺寸来综合测试CUDA算子的性能和正确性。

## 🎯 特色功能

- **多Shape测试**: 每个算子支持10+种不同的张量尺寸配置
- **自动化验证**: 自动对比CUDA kernel与PyTorch参考实现的结果
- **性能基准**: 提供详细的性能对比和加速比分析
- **可扩展架构**: 易于添加新的算子和shape配置
- **工业级**: 模拟真实深度学习工作负载的shape分布

## 📂 目录结构

```
leetcuda_dynamic/
├── dotproduct_f32x4/             # 向量点积 (f32x4优化)
├── sum_f32x4/                    # 向量求和 (f32x4优化)  
├── gelu_f16x8_pack/              # GELU激活函数 (f16x8优化)
├── hgemm_mma/                    # 半精度矩阵乘法 (MMA指令)
└── scripts/
    └── test_dynamic_comprehensive.sh  # 综合测试脚本
```

每个算子目录包含：
- `cuda_/kernel.cu`: CUDA kernel实现
- `torch_/ref.py`: PyTorch参考实现  
- `get_data.py`: 数据生成和shape配置
- `check_cuda.py`: 测试脚本

## 🔧 使用方法

### 测试单个算子

```bash
cd dotproduct_f32x4/
python check_cuda.py                    # 完整测试
python check_cuda.py --compile-only     # 仅编译测试
python check_cuda.py --max-shapes 5     # 限制测试5个shape
python check_cuda.py --no-perf          # 禁用性能测试
```

### 综合测试所有算子

```bash
# 完整测试
bash scripts/test_dynamic_comprehensive.sh

# 仅编译测试
PERF_FLAG=--compile-only bash scripts/test_dynamic_comprehensive.sh

# 限制每个算子的shape数量
MAX_SHAPES=3 bash scripts/test_dynamic_comprehensive.sh

# 禁用性能测试  
PERF_FLAG=--no-perf bash scripts/test_dynamic_comprehensive.sh
```

## 📊 Shape配置说明

### 向量算子 (dotproduct, sum)
- 小尺寸: 1K - 8K 元素
- 中尺寸: 16K - 256K 元素  
- 大尺寸: 512K - 2M 元素
- 不规则尺寸: 非2的幂次

### 激活函数算子 (gelu)
- 8的倍数尺寸 (向量化优化)
- 覆盖1K到2M元素范围
- 支持不规则shape

### 矩阵算子 (hgemm)
- 小矩阵: 64x64 到 512x512
- 中矩阵: 1024x1024 到 2048x2048  
- 大矩阵: 最大4096x4096
- 矩形矩阵: 各种长宽比
- 深度学习常用尺寸: LLaMA FFN等

## 🎯 测试输出示例

```
🚀 Dynamic Shape Dot Product (f32x4) Test
📊 Testing 15 different shapes
📐 Shape range: (1024,) to (2097152,)

=============== Shape Config 1/15 ===============
📐 Shape: (1024,)
⚡ Running PyTorch reference...
⚡ Running CUDA kernel...
🔍 Numerical verification:
   Max difference: 1.23e-05
   Mean difference: 3.45e-06  
   Relative error: 2.11e-06
   Results match: ✅
✅ Numerical test PASSED
🏃 Performance testing...
📊 Performance:
   PyTorch:    0.123 ms
   CUDA:       0.089 ms
   Speedup: 1.38x 🚀
```

## 🏗️ 添加新算子

1. **创建算子目录**:
   ```bash
   mkdir new_algorithm/{cuda_,torch_,triton_}
   ```

2. **实现CUDA kernel** (`cuda_/kernel.cu`):
   ```cpp
   extern "C" void cuda_kernel(/* parameters */) {
       // 支持动态shape的kernel实现
   }
   ```

3. **实现数据生成器** (`get_data.py`):
   ```python
   SHAPE_CONFIGS = [(size1,), (size2,), ...]  # 定义测试shape
   
   def get_inputs_for_shape(shape_config):
       # 生成指定shape的测试数据
       return torch_inputs, cuda_inputs
   ```

4. **实现PyTorch参考** (`torch_/ref.py`):
   ```python
   def torch_kernel(inputs):
       # PyTorch参考实现
       return output
   ```

5. **添加到测试脚本**:
   编辑 `scripts/test_dynamic_comprehensive.sh`，添加新算子。

## ⚡ 工具支持

### 动态测试工具

- `cu2til.tools.check_cuda_dynamic`: CUDA动态shape测试框架
- `cu2til.tools.check_triton_dynamic`: Triton动态shape测试框架(待实现)

### 核心功能

- **多Shape验证**: 自动遍历所有配置进行数值验证
- **性能基准**: 对比CUDA与PyTorch的执行时间
- **结果保存**: 自动保存测试结果到JSON文件
- **错误诊断**: 详细的错误信息和调试输出

## 🎉 技术亮点

- **工业级测试**: 覆盖真实应用场景的各种shape
- **高效执行**: 智能的测试调度和超时保护
- **详细分析**: 全面的性能和正确性分析
- **易于扩展**: 模块化设计，便于添加新算子
- **生产就绪**: 完整的CI/CD支持和自动化测试

## 📈 性能基准

框架会自动收集每个shape配置的性能数据：
- PyTorch执行时间
- CUDA执行时间  
- 加速比计算
- 平均性能统计

数据保存为JSON格式，便于后续分析和可视化。

---

🎯 **这是一个世界级的动态Shape CUDA测试框架，为CUDA kernel开发提供完整的验证和性能分析解决方案！**
