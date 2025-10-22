# Working Directory (work_dir) 使用指南

## 概述

NVGPU Server **完全支持**指定脚本执行的工作目录。本文档说明如何使用这个功能。

## 当前功能

### 1. 基础用法

```python
from server.nvgpu.client import NVGPUClient

client = NVGPUClient("http://localhost:8080")

# 在指定目录执行脚本
task_id = client.submit_task(
    script_path="/workspace/test/script.py",
    work_dir="/workspace/test",  # 脚本将在这个目录下执行
    args=["--arg1", "value"]
)
```

### 2. 工作目录的作用

当设置 `work_dir` 后：
- ✅ 脚本的当前工作目录 (CWD) 将是指定的目录
- ✅ 脚本中的相对路径将相对于这个目录
- ✅ 脚本可以访问该目录下的文件
- ✅ `os.getcwd()` 将返回这个目录

### 3. 实际执行

在 `task_runner.py` 中：
```python
process = subprocess.Popen(
    cmd,
    cwd=task.work_dir,  # ← 工作目录在这里设置
    env=env,
    stdout=stdout_file,
    stderr=stderr_file
)
```

## 常见使用场景

### 场景 1：脚本需要访问同目录下的文件

```python
# 目录结构：
# /workspace/my_test/
#   ├── test_script.py
#   ├── data.txt
#   └── config.json

# test_script.py 中使用相对路径：
# with open("data.txt") as f:
#     data = f.read()

# 提交任务：
client.submit_task(
    script_path="/workspace/my_test/test_script.py",
    work_dir="/workspace/my_test"  # 脚本可以访问 data.txt
)
```

### 场景 2：多个测试用例，各自独立

```python
test_cases = [
    "/workspace/tests/case1",
    "/workspace/tests/case2",
    "/workspace/tests/case3",
]

for test_dir in test_cases:
    task_id = client.submit_task(
        script_path=f"{test_dir}/test.py",
        work_dir=test_dir  # 每个测试在自己的目录下运行
    )
```

### 场景 3：llm_trans.py 的实际使用

```python
# 在 llm_trans.py 中：
task_id = client.submit_task(
    script_path=str((test_work_dir / "check_triton.py").absolute()),
    work_dir=str(test_work_dir.absolute()),  # 在测试目录下执行
    args=["--no-perf"] if args.no_perf else []
)

# 这样 check_triton.py 可以访问：
# - triton_/kernel.py
# - torch_/ref.py
# - get_data.py
# 等同目录下的文件
```

## 便捷方法（可选增强）

### 方案 A：自动使用脚本所在目录

创建一个辅助函数：

```python
from pathlib import Path

def submit_task_in_script_dir(client, script_path, **kwargs):
    """Submit task with work_dir automatically set to script's directory."""
    script_dir = str(Path(script_path).parent.absolute())
    return client.submit_task(
        script_path=script_path,
        work_dir=script_dir,
        **kwargs
    )

# 使用：
task_id = submit_task_in_script_dir(
    client,
    script_path="/workspace/test/script.py",  # 自动使用 /workspace/test 作为 work_dir
    args=["--arg1"]
)
```

### 方案 B：扩展 Client 类（推荐）

在 `client.py` 中添加：

```python
class NVGPUClient:
    # ... 现有方法 ...
    
    def submit_task_in_script_dir(
        self,
        script_path: str,
        task_type: str = "functional",
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        gpu_id: Optional[int] = None
    ) -> str:
        """Submit task with work_dir automatically set to script's directory.
        
        This is a convenience method that automatically uses the script's
        parent directory as the working directory.
        
        Args:
            script_path: Path to the Python script to run
            task_type: Task type ("functional" or "performance")
            args: Command line arguments for the script
            env: Environment variables
            gpu_id: Specific GPU ID, or None for auto-assignment
            
        Returns:
            Task ID
            
        Example:
            >>> # These are equivalent:
            >>> client.submit_task(
            ...     script_path="/workspace/test/script.py",
            ...     work_dir="/workspace/test"
            ... )
            >>> client.submit_task_in_script_dir(
            ...     script_path="/workspace/test/script.py"
            ... )
        """
        from pathlib import Path
        work_dir = str(Path(script_path).parent.absolute())
        
        return self.submit_task(
            script_path=script_path,
            task_type=task_type,
            work_dir=work_dir,
            args=args,
            env=env,
            gpu_id=gpu_id
        )
```

使用示例：
```python
# 方便！自动使用脚本所在目录
task_id = client.submit_task_in_script_dir(
    script_path="/workspace/test/check_triton.py"
)
```

### 方案 C：添加 use_script_dir 参数

修改 `submit_task` 方法：

```python
def submit_task(
    self,
    script_path: str,
    task_type: str = "functional",
    work_dir: str = ".",
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    gpu_id: Optional[int] = None,
    use_script_dir: bool = False  # ← 新参数
) -> str:
    """Submit a task to the server.
    
    Args:
        script_path: Path to the Python script to run
        task_type: Task type ("functional" or "performance")
        work_dir: Working directory for the script (ignored if use_script_dir=True)
        args: Command line arguments for the script
        env: Environment variables
        gpu_id: Specific GPU ID, or None for auto-assignment
        use_script_dir: If True, use script's directory as work_dir
        
    Returns:
        Task ID
    """
    # Auto-detect work_dir if requested
    if use_script_dir:
        from pathlib import Path
        work_dir = str(Path(script_path).parent.absolute())
    
    payload = {
        "script_path": script_path,
        "task_type": task_type,
        "work_dir": work_dir,
        "args": args or [],
    }
    
    # ... 其余代码不变 ...
```

使用：
```python
# 自动使用脚本所在目录
task_id = client.submit_task(
    script_path="/workspace/test/script.py",
    use_script_dir=True  # ← 简单明了
)

# 或明确指定
task_id = client.submit_task(
    script_path="/workspace/test/script.py",
    work_dir="/workspace/another/dir"  # ← 灵活控制
)
```

## 推荐实现

**推荐方案 B：添加 `submit_task_in_script_dir` 便捷方法**

优点：
- ✅ 不破坏现有 API
- ✅ 向后兼容
- ✅ 语义清晰
- ✅ 实现简单
- ✅ 易于理解和维护

## 验证工作目录

创建测试脚本验证：

```python
# test_work_dir.py
import os
import sys

print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {os.path.abspath(__file__)}")
print(f"Directory contents: {os.listdir('.')}")

# 尝试读取当前目录的文件
try:
    with open("data.txt") as f:
        print(f"Successfully read data.txt: {f.read()[:50]}")
except FileNotFoundError:
    print("data.txt not found in current directory")
```

测试：
```python
# 创建测试目录和文件
import os
os.makedirs("/tmp/test_dir", exist_ok=True)
with open("/tmp/test_dir/data.txt", "w") as f:
    f.write("Hello from data.txt!")

# 提交任务
task_id = client.submit_task(
    script_path="/tmp/test_dir/test_work_dir.py",
    work_dir="/tmp/test_dir"
)

# 等待完成并查看输出
result = client.get_task(task_id)
stdout = client.get_full_task_log(task_id, "stdout")
print(stdout)

# 应该看到：
# Current working directory: /tmp/test_dir
# Successfully read data.txt: Hello from data.txt!
```

## 常见问题

### Q: 为什么我的脚本找不到文件？

A: 确保设置了正确的 `work_dir`：
```python
# ❌ 错误：使用默认的 "."
client.submit_task(script_path="/workspace/test/script.py")

# ✅ 正确：明确指定工作目录
client.submit_task(
    script_path="/workspace/test/script.py",
    work_dir="/workspace/test"
)
```

### Q: 绝对路径 vs 相对路径？

A: 推荐使用绝对路径：
```python
from pathlib import Path

# ✅ 推荐：使用绝对路径
test_dir = Path("/workspace/test").absolute()
client.submit_task(
    script_path=str(test_dir / "script.py"),
    work_dir=str(test_dir)
)

# ⚠️  可行但不推荐：相对路径（相对于服务器的 CWD）
client.submit_task(
    script_path="test/script.py",
    work_dir="test"
)
```

### Q: 如何在 llm_trans.py 中使用？

A: 已经正确使用了：
```python
# cu2til/trans/dev_/llm_trans.py line 636-646
task_id = client.submit_task(
    script_path=str((test_work_dir / f"check_triton{CHECK_SUFFIX}.py").absolute()),
    task_type="functional",
    work_dir=str(test_work_dir.absolute()),  # ← 已经正确设置
    args=task_args,
    gpu_id=args.nvgpu_gpu
)
```

## 总结

- ✅ **当前功能完整**：通过 `work_dir` 参数完全支持指定工作目录
- ✅ **已在使用**：`llm_trans.py` 已正确使用此功能
- ✅ **可选增强**：可添加便捷方法使 API 更友好
- ✅ **向后兼容**：所有改进都保持向后兼容

## 相关文档

- [Client API](../client.py)
- [Task Runner](../task_runner.py)
- [llm_trans Integration](../../../cu2til/trans/dev_/LLM_TRANS_NVGPU_INTEGRATION.md)
