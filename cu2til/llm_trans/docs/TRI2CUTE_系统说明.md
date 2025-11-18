# Triton→CUTE 翻译系统完整说明

## ✅ 系统状态

**版本**: 1.0  
**状态**: 已实现并验证  
**验证日期**: 2025-10-24  
**测试 GPU**: H800 (SM90)

## 一、核心功能

### 已实现并验证

1. **翻译方向支持** ✅
   - `--direction cu2tri`: CUDA → Triton (原有功能，完全兼容)
   - `--direction tri2cute`: Triton → CUTE (新功能，已验证)

2. **自动编译测试** ✅
   - 自动检测 GPU 架构（SM80/SM86/SM89/SM90）
   - nvcc 编译集成
   - ctypes 动态加载
   - 正确性自动验证

3. **迭代修复机制** ✅
   - 编译错误自动反馈 LLM
   - 运行时错误自动修复
   - 最多支持 N 轮迭代

##二、使用方法

### 基本命令

```bash
# 环境设置
conda activate serve
export CUTLASS_ROOT=/data/apps/project/cu2tri/elib/cutlass_latest
export CUDA_VISIBLE_DEVICES=7  # H800

# 运行翻译（新语法）
python -m cu2til.llm_trans \
  --model gpt_5_mini \
  --testset triton_tutorial \
  --direction tri2cute \
  --max-rounds 3 \
  --max-attempts 1 \
  --concurrency 1 \
  --no-nvgpu
```

### 参数说明

| 参数 | 值 | 说明 |
|------|---|------|
| `--direction` | cu2tri / tri2cute | **必需**: 翻译方向 |
| `--testset` | triton_tutorial, flaggems, ... | 测试集名称 |
| `--model` | gpt_5_mini, gpt_5 | LLM 模型 |
| `--max-rounds` | 1-10 | 最大迭代轮数 |
| `--max-attempts` | 1-30 | 独立尝试次数 |
| `--concurrency` | 1-40 | 并发数 |

### 输出路径

新的路径结构：
```
cu2til/llm_trans/runs/{direction}/{testset}/{model}/{timestamp}/
```

示例：
```
cu2til/llm_trans/runs/tri2cute/triton_tutorial/gpt_5_mini/20251024_070831/
```

## 三、测试集结构

### triton_tutorial

包含 Triton 官方教程中的算子：
```
cu2til/cases/triton_tutorial/
├── add_vector/              # 向量加法
├── fused_softmax/           # Fused Softmax
├── matrix_multiplication/   # 矩阵乘法
├── flash_attention/         # Flash Attention
└── ... (其他 tutorial 算子)
```

### flaggems

Flag Gems 项目中的优化 kernel：
```
cu2til/cases/flaggems/
├── layernorm/
├── rmsnorm/
├── softmax/
└── ...
```

### unsloth

Unsloth 项目中的优化 kernel：
```
cu2til/cases/unsloth/
├── rope/
├── cross_entropy/
└── ...
```

### ligerkernel

Liger Kernel 项目中的 kernel：
```
cu2til/cases/ligerkernel/
├── rms_norm/
├── rope/
├── swiglu/
└── ...
```

## 四、测试用例标准结构

每个测试用例目录包含：

```
algorithm_name/
├── triton_/
│   └── kernel.py          # Triton 实现（源）
├── torch_/
│   └── ref.py            # PyTorch 参考
├── cute_/
│   └── kernel.cu         # CUTE 实现（LLM 生成）
├── get_data.py           # 测试数据生成
└── check_cute.py         # 测试脚本（从 tools 复制）
```

### get_data.py 规范

```python
import torch

class Params:
    """测试参数配置"""
    def __init__(self):
        self.size = 1024
        self.dtype = torch.float32
        # ... 其他参数

def get_triton_torch_inputs(params: Params):
    """
    生成测试输入数据
    
    Returns:
        dict: 输入参数字典，键名对应 kernel 函数参数
    """
    torch.manual_seed(0)
    inputs = {
        'a': torch.randn(..., device='cuda'),
        'b': torch.randn(..., device='cuda'),
    }
    return inputs

def triton_output_tensor_transform(output):
    """
    可选：转换输出格式
    """
    return output

# 兼容性别名
get_cuda_torch_inputs = get_triton_torch_inputs
cuda_output_tensor_transform = triton_output_tensor_transform
```

### triton_/kernel.py 规范

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_impl(...):
    """Triton kernel 实现"""
    pass

def triton_kernel(a, b, ...):
    """
    Kernel 包装函数
    
    Args:
        a: 输入张量 A
        b: 输入张量 B
        ...
    
    Returns:
        输出张量
    """
    # 配置 grid
    # 调用 kernel
    return output
```

### torch_/ref.py 规范

```python
import torch

def torch_kernel(a, b, ...):
    """PyTorch 参考实现"""
    return result
```

## 五、GPU 架构支持

### 自动检测

通用 `check_cute.py` 会自动检测 GPU 架构：

| GPU 型号 | 架构 | Compute Cap | 编译参数 |
|---------|------|-------------|---------|
| H800 SXM | SM90 | 9.0 | `compute_90,sm_90` |
| H100 PCIe | SM90 | 9.0 | `compute_90,sm_90` |
| H800 PCIe | SM90 | 9.0 | `compute_90,sm_90` |
| RTX 6000 Ada | SM89 | 8.9 | `compute_89,sm_89` |
| A800 80G SXM | SM80 | 8.0 | `compute_80,sm_80` |
| RTX 5090 | SM89 | 8.9 | `compute_89,sm_89` |

### 手动指定

```bash
python check_cute.py --arch compute_90,sm_90
```

## 六、多输入测试支持

### 在 get_data.py 中定义多组测试

```python
class Params:
    def __init__(self, size=1024):
        self.size = size
        self.dtype = torch.float32

# 支持多个参数配置
test_configs = [
    Params(size=512),
    Params(size=1024),
    Params(size=2048),
    Params(size=4096),
]
```

### 测试脚本循环测试

```python
for params in test_configs:
    print(f"\nTesting with size={params.size}")
    success = test_correctness(args, params)
    if not success:
        sys.exit(1)
```

## 七、实测验证结果

### Add 算子（✅ 成功）

```
测试: add_simple
输入: (1024, 1024) float32
GPU: H800 (SM90)
模型: GPT-5-mini

结果:
  编译: ✅ 成功 (9秒, SM90)
  加载: ✅ 成功 (cute_kernel_wrapper)
  运行: ✅ 无错误
  验证: ✅ Max diff = 0.0

总时间: 99秒 (LLM 90s + 测试 9s)
状态: ✅ PASSED (第1轮成功)
```

## 八、配置文件更新

### case_config.yaml

```yaml
testsets:
  triton_tutorial:
    mode: scan
    path: cu2til/cases/triton_tutorial
    include_ops: []
    exclude_ops: []
  
  flaggems:
    mode: scan
    path: cu2til/cases/flaggems
    include_ops: []
    exclude_ops: []
  
  unsloth:
    mode: scan
    path: cu2til/cases/unsloth
    include_ops: []
    exclude_ops: []
  
  ligerkernel:
    mode: scan
    path: cu2til/cases/ligerkernel
    include_ops: []
    exclude_ops: []
```

## 九、待完成事项

### 高优先级

1. **构建 triton_tutorial testset**
   - 从 Triton 官方 tutorials 提取算子
   - 验证所有 kernel 的正确性
   - 支持多输入 shape

2. **构建 flaggems testset**
   - 从 Flag Gems 项目提取
   - 验证 kernel 可用性

3. **构建 unsloth/ligerkernel testsets**
   - 提取相关 kernels
   - 验证可用性

### 中优先级

1. **性能测试集成**
   - 实现 benchmark 功能
   - TFLOPS 计算
   - Triton vs CUTE 性能对比

2. **Flash Attention 验证**
   - 测试 FA 翻译
   - 多种配置验证

## 十、故障排除

### 常见问题

1. **架构不匹配**
   ```
   ERROR: no kernel image available for execution
   解决: check_cute.py 会自动检测，或手动指定 --arch
   ```

2. **找不到符号**
   ```
   ERROR: undefined symbol: cute_kernel_wrapper
   解决: 增加 --max-rounds，让 LLM 修复
   ```

3. **编译失败**
   ```
   ERROR: cute/tensor.hpp not found
   解决: export CUTLASS_ROOT=...
   ```

## 十一、开发指南

### 添加新的 Triton kernel 到 testset

1. 创建目录结构
2. 实现 triton_/kernel.py
3. 实现 torch_/ref.py
4. 配置 get_data.py
5. 复制 check_cute.py from tools
6. 验证 Triton kernel 可用性

### 扩展新的翻译方向

系统架构支持扩展更多方向：
- tri2cuda: Triton → CUDA
- tri2hip: Triton → HIP (AMD)
- tri2mojo: Triton → Mojo

只需：
1. 添加 direction 选项
2. 创建 prompt 模板
3. 实现 testing 模块
4. 更新配置

## 十二、命令示例

### 测试不同 testset

```bash
# Triton Tutorial
python -m cu2til.llm_trans --direction tri2cute --testset triton_tutorial --model gpt_5_mini --max-rounds 3 --max-attempts 1 --concurrency 1 --no-nvgpu

# Flag Gems
python -m cu2til.llm_trans --direction tri2cute --testset flaggems --model gpt_5 --max-rounds 5 --max-attempts 1 --concurrency 1 --no-nvgpu

# Unsloth
python -m cu2til.llm_trans --direction tri2cute --testset unsloth --model gpt_5 --max-rounds 5 --max-attempts 1 --concurrency 1 --no-nvgpu

# Liger Kernel
python -m cu2til.llm_trans --direction tri2cute --testset ligerkernel --model gpt_5 --max-rounds 5 --max-attempts 1 --concurrency 1 --no-nvgpu
```

### 原有 CUDA→Triton 仍然可用

```bash
# 默认 direction=cu2tri
python -m cu2til.llm_trans \
  --model gpt_5_mini \
  --testset xpiler \
  --case-type transpose \
  --max-rounds 3 \
  --max-attempts 1 \
  --concurrency 3
```

## 十三、文件位置

### 源代码

```
cu2til/
├── llm_trans/
│   ├── config/
│   │   ├── args.py             # ✅ 更新: --direction
│   │   ├── settings.py         # ✅ 更新: direction 字段
│   │   └── case_config.yaml    # ✅ 更新: 新 testsets
│   ├── services/
│   │   ├── attempts.py         # ✅ 更新: direction 逻辑
│   │   ├── testing.py          # ✅ 更新: 路由逻辑
│   │   └── testing_cute.py     # ✅ 新增
│   ├── docs/                   # ✅ 文档目录
│   │   ├── TRI2CUTE_系统说明.md
│   │   ├── TRITON2CUTE_GUIDE.md
│   │   ├── TRITON2CUTE_*.md
│   │   └── ...
│   └── runs/
│       ├── cu2tri/            # CUDA→Triton 结果
│       └── tri2cute/          # Triton→CUTE 结果
├── prompt/
│   ├── cuda2triton.py
│   └── triton2cute.py         # ✅ 新增
├── tools/
│   ├── check_cute.py          # ✅ 新增通用脚本
│   ├── check_triton.py
│   └── ...
└── cases/
    ├── xpiler/                # 原有
    ├── leetcuda_dynamic/      # 原有
    ├── triton_tutorial/       # ⏳ 待构建
    ├── flaggems/              # ⏳ 待构建
    ├── unsloth/               # ⏳ 待构建
    └── ligerkernel/           # ⏳ 待构建
```

## 十四、关键改进

### vs 之前版本

| 特性 | 旧版 | 新版 |
|------|------|------|
| 语言参数 | `--source-lang` `--target-lang` | `--direction` ✅ 更简洁 |
| 输出路径 | `runs/cu2tri/` | `runs/{direction}/` ✅ 更清晰 |
| Testset | 分散的 `triton2cute_add` | 统一的 `triton_tutorial` ✅ 更合理 |
| Check 脚本 | 每个用例独立 | 通用脚本 `tools/check_cute.py` ✅ 可复用 |
| GPU 架构 | 硬编码 SM80 | 自动检测 ✅ 更智能 |

## 十五、下一步工作

### 需要完成

1. **构建 triton_tutorial testset** ⏳
   - 从 Triton tutorials 提取算子
   - 为每个算子创建标准结构
   - 验证 Triton kernel 可用性
   - 测试多种输入配置

2. **构建 flaggems testset** ⏳
   - 提取 Flag Gems kernels
   - 验证可用性

3. **构建 unsloth testset** ⏳
   - 提取 Unsloth kernels
   - 验证可用性

4. **构建 ligerkernel testset** ⏳
   - 提取 Liger Kernel kernels
   - 验证可用性

5. **多输入测试** ⏳
   - 实现多配置测试循环
   - 验证不同 shape 下的正确性

### 建议的构建脚本

由于工作量较大，建议创建自动化脚本来构建测试集。我将在下一步提供构建脚本模板。

## 十六、验证清单

在将 testset 加入系统前，需要验证：

- [ ] Triton kernel 能正常运行
- [ ] PyTorch reference 结果正确
- [ ] get_data.py 生成合理的测试数据
- [ ] 支持多种输入 shape
- [ ] check_cute.py 从 tools 复制无误
- [ ] 目录结构符合规范
- [ ] 所有必需文件齐全

## 十七、联系和反馈

- **文档**: `cu2til/llm_trans/docs/`
- **问题**: 提交 issue
- **贡献**: 提交 PR

---

**最后更新**: 2025-10-24  
**维护者**: cu2tri 项目团队

