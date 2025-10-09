# 任务参数测试说明

## 概述

这些测试验证 NVGPU 服务器的任务参数（`task_type`、`task_mode`、`task_label`）在各种组合下都能正常工作。

## 测试文件

### 1. `example_task_parameters.py` - 完整测试

**包含 15 个测试用例**，覆盖所有可能的参数组合：

```bash
python /workspace/server/nvgpu/examples/example_task_parameters.py
```

**测试场景：**
1. 只有 `task_mode='shared'`
2. 只有 `task_mode='exclusive'`
3. 只有 `task_type='functional'` (测试智能默认 → shared)
4. 只有 `task_type='performance'` (测试智能默认 → exclusive)
5. 只有 `task_type='both'` (测试智能默认 → exclusive)
6. 只有 `task_label`
7. `task_type='functional'` + `task_label`
8. `task_type='performance'` + `task_label`
9. `task_type='functional'` + `task_mode='exclusive'` (覆盖智能默认)
10. `task_type='performance'` + `task_mode='shared'` (覆盖智能默认)
11. 所有参数：`type='functional'` + `mode='shared'` + `label`
12. 所有参数：`type='performance'` + `mode='exclusive'` + `label`
13. 无任何参数（全部使用默认值）
14. `task_type='both'` + `task_label`
15. `task_mode='exclusive'` + `task_label`（无 type）

### 2. `example_task_parameters_quick.py` - 快速测试

**包含 12 个核心测试用例**，快速验证关键功能：

```bash
python /workspace/server/nvgpu/examples/example_task_parameters_quick.py
```

**适合：**
- 快速验证服务器功能
- CI/CD 集成
- 快速回归测试

## 验证的功能

### ✅ 智能默认值

| task_type | 默认 task_mode |
|-----------|---------------|
| `functional` | `shared` |
| `performance` | `exclusive` |
| `both` | `exclusive` |
| (无) | `shared` |

### ✅ 参数可选性

- ✓ 所有参数都是可选的
- ✓ 可以只指定任意一个参数
- ✓ 可以指定任意组合
- ✓ 完全不指定也能正常工作

### ✅ 显式覆盖

- ✓ `task_mode` 可以覆盖基于 `task_type` 的智能默认值
- ✓ 显式指定的值优先级最高

### ✅ 参数独立性

- ✓ `task_label` 不影响 `task_mode` 的默认值
- ✓ 每个参数可以独立使用

## 预期结果

所有测试都应该：
1. ✓ 成功提交任务
2. ✓ 使用正确的 `task_mode`（匹配智能默认值或显式指定）
3. ✓ 任务成功完成（status=completed, exit_code=0）

## 输出示例

```
╔══════════════════════════════════════════════════════════════════╗
║         Task Parameters Test (type, mode, label)                 ║
╚══════════════════════════════════════════════════════════════════╝

Running tests...

──────────────────────────────────────────────────────────────────
✓ 1. Only task_mode='shared'
   Task ID: a1b2c3d4...
   Expected mode: shared
   Actual mode:   shared ✓
   Status: completed
   Parameters: task_mode='shared'

✓ 3. Only task_type='functional' (default: shared)
   Task ID: e5f6g7h8...
   Expected mode: shared
   Actual mode:   shared ✓
   Status: completed
   Parameters: task_type='functional'

✓ 9. task_type='functional' + task_mode='exclusive' (override)
   Task ID: i9j0k1l2...
   Expected mode: exclusive
   Actual mode:   exclusive ✓
   Status: completed
   Parameters: task_type='functional', task_mode='exclusive'

──────────────────────────────────────────────────────────────────

📊 Summary

Total: 15 tests
Passed: 15 tests ✓
Failed: 0 tests ✗

🎉 All tests passed!

──────────────────────────────────────────────────────────────────

📋 Smart Defaults Verification

Task type → Default mode:
  ✓ functional   → shared
  ✓ performance  → exclusive
  ✓ both         → exclusive

──────────────────────────────────────────────────────────────────

📋 Override Verification

Can task_mode override smart defaults?
  ✓ functional + exclusive override      → exclusive
  ✓ performance + shared override        → shared

╚══════════════════════════════════════════════════════════════════╝
```

## 使用场景

### 场景 1: 功能测试（默认共享）

```python
client.submit_task(
    script_path="test.py",
    task_type="functional"
)
# → 自动使用 shared 模式
```

### 场景 2: 性能测试（默认独占）

```python
client.submit_task(
    script_path="benchmark.py",
    task_type="performance"
)
# → 自动使用 exclusive 模式
```

### 场景 3: 显式控制

```python
client.submit_task(
    script_path="test.py",
    task_type="functional",
    task_mode="exclusive"  # 覆盖默认的 shared
)
# → 使用 exclusive 模式（即使 functional 通常是 shared）
```

### 场景 4: 添加标签

```python
client.submit_task(
    script_path="test.py",
    task_type="functional",
    task_label="xpiler_cuda/add_3_3_256/cuda_vs_triton"
)
# → shared 模式 + 标签用于识别
```

### 场景 5: 最简单的方式

```python
client.submit_task(script_path="test.py")
# → 全部使用默认值（shared 模式）
```

## 故障排除

### 如果测试失败

1. **检查服务器是否运行**
   ```bash
   curl http://localhost:8080/health
   ```

2. **检查 GPU 是否可用**
   ```bash
   curl http://localhost:8080/gpus
   ```

3. **查看服务器日志**
   ```bash
   tail -f /workspace/server/nvgpu/logs/nvgpu_server.log
   ```

4. **验证测试脚本存在**
   ```bash
   ls -la /workspace/server/nvgpu/test_scripts/simple_functional_test.py
   ```

## 相关文档

- `API_REFERENCE_ZH.md` - API 参考文档
- `README.md` - 服务器总览
- `QUICKSTART.md` - 快速入门指南

