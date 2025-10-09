# NVGPU Server 文档中心

**GPU 任务调度服务器完整文档**

---

## 🚀 快速导航

### 新手入门
- **[快速入门](QUICKSTART.md)** ⭐⭐⭐ - 5分钟上手指南

### 核心文档
- **[设计文档](DESIGN.md)** ⭐⭐⭐⭐ - 完整设计说明（强烈推荐）
- **[变更说明](CHANGES.md)** ⭐⭐⭐ - 版本变更详情
- **[API 参考手册](API_REFERENCE.md)** ⭐⭐⭐ - 完整的 API 文档

### 配置与参考
- **[GPU 配置指南](GPU_CONFIG_GUIDE.md)** - GPU 资源配置
- **[工作目录指南](WORK_DIR_GUIDE.md)** - work_dir 功能
- **[目录结构说明](DIRECTORY_STRUCTURE.md)** - 文件组织

### 其他
- **[变更日志](CHANGELOG.md)** - 版本历史
- **[待办事项](TODO.md)** - 开发计划

---

## 📦 核心特性

### 概念清晰分离

```python
client.submit_task(
    "test.py",
    task_mode="shared",      # GPU 行为控制（智能默认）
    task_type="functional",  # 业务分类（可选）
    task_label="xpiler_cuda/add_3_3_256/cuda_vs_triton"  # 具体标识（可选）
)
```

#### **task_mode** - GPU 行为控制
- **用途:** 控制 GPU 如何执行任务
- **取值:** `"exclusive"` 或 `"shared"`
- **默认:** 智能默认（基于 `task_type`）

#### **task_type** - 业务分类
- **用途:** 业务层面的分类，用于统计和筛选
- **取值:** `"functional"`, `"performance"`, `"both"`
- **默认:** `None`（可选）

#### **task_label** - 具体标识
- **用途:** 具体的测试标签，精确识别
- **取值:** 任意字符串（建议层次结构）
- **默认:** `None`（可选）

---

## 💡 快速示例

### 最简单（90% 场景）
```python
client.submit_task("test.py")
```

### 功能测试
```python
client.submit_task("test.py", task_type="functional")
```

### 性能测试（自动 exclusive）
```python
client.submit_task("benchmark.py", task_type="performance")
```

### 完整标识
```python
client.submit_task(
    "test.py",
    task_type="functional",
    task_label="xpiler_cuda/add_3_3_256/cuda_vs_triton"
)
```

---

## 📊 智能默认值

```
task_type="functional"   → task_mode="shared"     ✓
task_type="performance"  → task_mode="exclusive"  ✓
task_type="both"         → task_mode="exclusive"  ✓ (包含性能测试)
task_type=None           → task_mode="shared"     ✓
```

---

## 🎯 设计优势

### 1. 概念清晰
- **task_mode**: 技术层（GPU 如何执行）
- **task_type**: 业务层（测试分类）
- **task_label**: 标识层（精确识别）

### 2. 灵活性强
- 所有参数可选
- 智能默认减少输入
- 按需指定任何参数

### 3. 易用性好
- 90% 场景 1 个参数
- 5% 场景 2 个参数
- 3% 场景 3 个参数

### 4. 可扩展
- `task_type` 可添加新类型
- `task_label` 支持任意字符串
- 不破坏现有代码

---

## 🗂️ 文档结构

```
docs/
├── README.md                   # 本文件 - 文档索引
├── QUICKSTART.md              # 快速入门
├── DESIGN.md                  # 设计文档 ⭐⭐⭐⭐
├── CHANGES.md                 # 变更说明
├── API_REFERENCE.md           # API 参考 ⭐⭐⭐
├── GPU_CONFIG_GUIDE.md        # GPU 配置指南
├── WORK_DIR_GUIDE.md          # 工作目录指南
├── DIRECTORY_STRUCTURE.md     # 目录结构说明
├── CHANGELOG.md               # 变更日志
└── TODO.md                    # 待办事项
```

---

## 📚 推荐阅读路径

### 新手
1. [快速入门](QUICKSTART.md) - 5分钟上手
2. [设计文档](DESIGN.md) - 理解核心概念
3. [API 参考](API_REFERENCE.md) - 查看详细 API

### 进阶用户
1. [GPU 配置指南](GPU_CONFIG_GUIDE.md) - 自定义配置
2. [工作目录指南](WORK_DIR_GUIDE.md) - 高级功能
3. [变更说明](CHANGES.md) - 版本变更

### 开发者
1. [目录结构说明](DIRECTORY_STRUCTURE.md) - 代码组织
2. [变更日志](CHANGELOG.md) - 版本历史
3. [待办事项](TODO.md) - 开发计划

---

## 🎓 示例代码

完整示例请参考：
- `/workspace/server/nvgpu/examples/` - Python 示例
- 各文档中的代码片段

---

## 💬 获取帮助

遇到问题？
1. 查看 [快速入门](QUICKSTART.md)
2. 阅读 [API 参考](API_REFERENCE_ZH.md)
3. 检查 [变更日志](CHANGELOG.md)

---

## 🔄 版本历史

- **v2.5** (2025-01) - 概念分离设计 + 智能默认值
- **v2.0** (2024-12) - 任务驱动 GPU 模式
- **v1.3** (2024-11) - 日志系统增强
- **v1.2** (2024-11) - 并发控制优化
- **v1.0** (2024-10) - 初始版本

完整历史请参考 [变更日志](CHANGELOG.md)。

---

**最后更新:** 2025-01-15  
**版本:** v2.5  
**返回:** [主 README](../README.md)
