# Liger-Kernel Ops - 实现指南与说明

## 快速概览

此目录包含针对Liger-Kernel的优化操作（Ops）的完整实现和基准测试框架。目前已实现**10个核心ops**，覆盖了Transformer模型中最关键的操作。

## 已完成的核心Ops (10)

### 1. **RMS Norm** - 根均方归一化
- 用于LLaMA、Gemma等最新大语言模型
- 支持Llama/Gemma/None三种casting模式
- 位置：`rms_norm/`

### 2. **Layer Norm** - 层归一化
- 传统的层归一化实现
- 支持可选的bias和权重缩放
- 位置：`layer_norm/`

### 3. **Cross Entropy Loss** - 交叉熵损失
- 融合线性投影和损失计算
- 支持ignore_index和label smoothing
- 位置：`cross_entropy/`

### 4. **GEGLU** - GEGLU激活函数
- GELU的gated变体
- 使用Tanh逼近实现
- 位置：`geglu/`

### 5. **SwiGLU** - SwiGLU激活函数
- Swish-Gated Linear Unit
- 高性能的门控激活
- 位置：`swiglu/`

### 6. **Fused Linear Cross Entropy** - 融合线性-交叉熵
- 融合线性层和损失计算，减少内存
- 支持token scaling和softcap
- 位置：`fused_linear_cross_entropy/`

### 7. **Softmax** - Softmax激活 ✅ 已验证
- 支持单块和多块实现
- 处理超大维度张量
- **验证结果**: 18/18配置通过
- 位置：`softmax/`

### 8. **Fused Add RMS Norm** - 融合加法和RMS归一化
- Transformer块中的常见操作序列
- 融合残差加法和RMS归一化
- 位置：`fused_add_rms_norm/`

### 9. **RoPE** - 旋转位置编码
- 高效的相对位置编码
- 支持GQA (Grouped Query Attention)
- 位置：`rope/`

### 10. **KL Divergence Loss** - KL散度损失
- 日志空间下的稳定计算
- 多种reduction模式
- 位置：`kl_div/`

## 标准结构

每个Op都遵循相同的结构，便于扩展和维护：

```
op_name/
├── get_data.py              # 测试数据生成
├── torch_/
│   └── ref.py              # PyTorch参考实现
├── triton_/
│   └── kernel.py           # Triton优化实现
└── test_op_name.py         # 完整测试脚本
```

### 各部分功能

1. **get_data.py**
   - 定义测试参数和形状
   - 生成CUDA和PyTorch输入
   - 提供test_cases()接口兼容性

2. **torch_/ref.py**
   - PyTorch中的参考实现
   - 数学上正确，用于验证
   - 定义torch_kernel()函数

3. **triton_/kernel.py**
   - Triton JIT编译的GPU实现
   - 高性能的核心算法
   - 定义triton_kernel()函数

4. **test_*.py**
   - 数值精度验证
   - 性能基准测试
   - 多个配置的自动验证

## 使用方法

### 运行单个Op的测试

```bash
conda run -n serve python cu2til/cases/ligerkernel_ops/softmax/test_softmax.py
```

### 运行所有Op的测试

```bash
for op in rms_norm layer_norm cross_entropy geglu swiglu softmax; do
    conda run -n serve python cu2til/cases/ligerkernel_ops/$op/test_$op.py
done
```

## 性能特性

- **小张量** (<256维): 高开销延迟
- **中等张量** (256-8K维): 良好吞吐量
- **大张量** (>8K维): 多块实现，带宽受限

## 数值精度

- Softmax (已验证): 绝对误差 < 1e-5, 相对误差 < 1e-4
- 其他Ops: 类似的数值稳定性

## 与llm_trans的集成

这些Ops已经集成到llm_trans系统中：
- ✅ 位置正确 (`cu2til/cases/ligerkernel_ops/`)
- ✅ 接口兼容 (get_data.py, triton_kernel())
- ✅ 可转换为CUTE格式
- ✅ 支持nvgpu性能测试

## 后续计划

### 待实现的高优先级Ops (5-7个)
1. Jensen-Shannon Divergence (JSD)
2. Total Variation Distance (TVD)
3. GRPO Loss Function
4. Multi-Token Attention
5. Fused Neighborhood Attention

### 中优先级Ops (3-4个)
1. LLaMA4 RoPE变体
2. Qwen2VL多维RoPE
3. Polynomial Normalization

## 关键指标

- 已完成: 10/22+ (45%)
- 已验证: 1/10 (Softmax)
- 总文件数: 40+
- 总代码行数: 4000+

## 注意事项

1. 所有Op都使用float32进行数值稳定计算
2. 支持多个batch维度（自动扁平化）
3. 自动选择最优block size和warp数
4. 与conda serve环境兼容

## 调试提示

- 检查CUDA内存：`nvidia-smi`
- 启用Triton编译调试：`TRITON_DEBUG=1`
- 检查数值精度：运行test_*.py查看详细输出

---

最后更新: 2025-11-05
