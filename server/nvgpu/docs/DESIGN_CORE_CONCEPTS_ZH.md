# NVGPU 核心设计概念说明

本文档用简洁的中文解释 NVGPU Server 的核心设计逻辑。

[完整技术文档请参考: DESIGN_ARCHITECTURE.md (English)]

---

## 核心概念澄清

### ⚠️ 最容易混淆的概念

很多用户会混淆以下两个**完全独立**的概念:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  概念1: GPU 模式 (exclusive/shared)                          │
│  ────────────────────────────────────────────────────────── │
│  作用: 控制一个GPU能同时运行多少个任务                         │
│  设置者: 管理员通过API手动设置                                │
│  影响: 任务调度逻辑                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  概念2: 任务类型 (functional/performance)                    │
│  ────────────────────────────────────────────────────────── │
│  作用: 标注任务的用途(仅标签,不影响调度)                       │
│  设置者: 用户提交任务时指定                                   │
│  影响: 无(仅用于日志分类和统计)                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 关键点

**GPU模式切换是手动的,不是自动的!**

- ❌ 错误理解: "提交performance任务时GPU自动切换到exclusive模式"
- ✅ 正确理解: "管理员先设置GPU为exclusive模式,然后用户提交任务"

- ❌ 错误理解: "任务类型决定GPU模式"
- ✅ 正确理解: "GPU模式和任务类型完全独立,互不影响"

---

## GPU 模式详解

### 1. Exclusive 模式 (独占模式)

**含义:** GPU同一时刻只能运行1个任务

**适用场景:**
- 性能基准测试 (需要稳定环境)
- 大显存任务 (需要全部VRAM)
- 需要可预测性能的任务

**行为:**
```python
# models.py Line 25-27
if self.mode == GPUMode.EXCLUSIVE:
    return len(self.running_tasks) == 0  # 只有没任务时才接受新任务
```

**示例:**
```
GPU 0: mode=EXCLUSIVE

时间     正在运行的任务      新任务提交        调度器行为
──────────────────────────────────────────────────────────
10:00   []                 T1提交           ✓ T1立即运行
10:01   [T1]               T2提交           ✗ T2进入队列等待
10:02   [T1]               T3提交           ✗ T3进入队列等待
10:05   []                 (T1完成)         ✓ T2开始运行
10:08   [T2]               -                -
10:10   []                 (T2完成)         ✓ T3开始运行
```

### 2. Shared 模式 (共享模式)

**含义:** GPU可以同时运行多个任务

**限制条件:**
1. `max_concurrent_tasks`: 最大并发任务数 (例如: 3个)
2. `memory_threshold`: 显存使用率阈值 (例如: 75%)

**适用场景:**
- 功能性测试 (轻量级)
- 批量测试 (追求吞吐量)
- 开发测试流程

**行为:**
```python
# models.py Line 31-36
# SHARED 模式检查
if len(self.running_tasks) >= self.max_concurrent_tasks:
    return False  # 限制1: 任务数达到上限
    
return self.current_memory_usage < self.memory_threshold  # 限制2: 显存未超标
```

**示例:**
```
GPU 0: mode=SHARED, max_tasks=3

时间     正在运行的任务      新任务提交        调度器行为
──────────────────────────────────────────────────────────
10:00   []                 T1,T2,T3提交     ✓ T1,T2,T3同时运行
10:01   [T1,T2,T3]         T4提交           ✗ T4等待(已有3个任务)
10:02   [T1,T2,T3]         T5提交           ✗ T5等待(已有3个任务)
10:05   [T2,T3]            (T1完成)         ✓ T4开始运行
10:06   [T2,T3,T4]         -                -
10:07   [T3,T4]            (T2完成)         ✓ T5开始运行
```

---

## 模式切换机制

### 何时切换?

**只在管理员明确调用API时切换!**

```python
# 示例: 通过API切换模式
import requests

# 切换到独占模式
response = requests.put(
    "http://localhost:8080/gpus/0/mode",
    json={"mode": "exclusive"}
)

# 切换到共享模式
response = requests.put(
    "http://localhost:8080/gpus/0/mode",
    json={"mode": "shared"}
)
```

或使用客户端:
```python
from server.nvgpu.client import NVGPUClient

client = NVGPUClient("http://localhost:8080")

# 切换模式
client.set_gpu_mode(gpu_id=0, mode="exclusive")
```

### 切换时正在运行的任务怎么办?

**关键设计: 正在运行的任务不受影响!**

```python
# gpu_manager.py Line 100-108
def set_gpu_mode(self, gpu_id: int, mode: GPUMode) -> bool:
    with self.lock:
        self.gpus[gpu_id].mode = mode  # 只改mode
        # 注意: 不修改 running_tasks !
        return True
```

**为什么这样设计?**
- ✅ 安全: 不中断正在运行的任务
- ✅ 数据完整性: 任务可以正常完成
- ✅ 不浪费计算: 已经运行的工作不会白费
- ✅ 可预测: 行为清晰明确

---

## 典型场景详解

### 场景1: Shared → Exclusive (有任务运行时切换)

```
时间线:
──────────────────────────────────────────────────────────────
10:00  │ GPU 0: mode=SHARED, running=[T1, T2, T3]
       │ (3个任务正在运行)
       │
10:01  │ 管理员: client.set_gpu_mode(0, "exclusive")
       │ ✓ 模式立即改变
       │ ✓ T1, T2, T3继续运行(不被中断)
       │
10:02  │ 用户: 提交任务T4
       │ 调度器检查: can_accept_task()
       │   → mode == EXCLUSIVE
       │   → len(running_tasks) == 3 > 0
       │   → return False ✗
       │ ✓ T4进入等待队列
       │
10:05  │ T1完成
       │ running=[T2, T3]
       │ 调度器: can_accept_task() → 还有2个任务 → False
       │ ✓ T4继续等待
       │
10:08  │ T2完成
       │ running=[T3]
       │ 调度器: can_accept_task() → 还有1个任务 → False
       │ ✓ T4继续等待
       │
10:10  │ T3完成
       │ running=[]
       │ 调度器: can_accept_task() → 没有任务了 → True ✓
       │ ✓ T4开始运行!
──────────────────────────────────────────────────────────────
```

**关键点:**
- 旧任务(T1,T2,T3)不受影响,正常完成
- 新任务(T4)遵守新规则(exclusive),必须等所有任务完成

### 场景2: Exclusive → Shared (有任务运行时切换)

```
时间线:
──────────────────────────────────────────────────────────────
10:00  │ GPU 0: mode=EXCLUSIVE, running=[T1]
       │
10:01  │ 管理员: client.set_gpu_mode(0, "shared", max_tasks=3)
       │ ✓ 模式立即改变
       │ ✓ T1继续运行
       │
10:02  │ 用户: 提交任务T2
       │ 调度器检查: can_accept_task()
       │   → mode == SHARED
       │   → len(running) == 1 < 3 ✓
       │   → memory_usage < 0.75 ✓
       │   → return True ✓
       │ ✓ T2立即开始运行(与T1并行)
       │
10:03  │ 用户: 提交任务T3, T4
       │ running=[T1, T2]
       │ ✓ T3开始运行(与T1,T2并行)
       │ running=[T1, T2, T3]
       │ ✗ T4进入队列(已达到max_tasks=3)
       │
10:05  │ T1完成
       │ running=[T2, T3]
       │ ✓ T4开始运行
──────────────────────────────────────────────────────────────
```

**关键点:**
- 旧任务(T1)继续运行
- 新任务(T2,T3,T4)可以并行运行(shared模式允许)

### 场景3: 任务类型不影响调度

```
场景: GPU设置为SHARED模式
──────────────────────────────────────────────────────────────
用户A: 提交 performance 类型任务T1
用户B: 提交 functional 类型任务T2
用户C: 提交 performance 类型任务T3

结果: T1, T2, T3都按照SHARED模式调度,可以并行运行
      任务类型(performance/functional)不影响调度逻辑!
──────────────────────────────────────────────────────────────

如果要让某个任务独占GPU,必须:
步骤1: 管理员设置 GPU mode = EXCLUSIVE
步骤2: 用户提交任务(任务类型随意)
```

---

## 任务类型说明

### 作用

任务类型**仅用于标记和分类**,完全不影响调度逻辑!

```python
# models.py Line 13-15
class TaskType(str, Enum):
    FUNCTIONAL = "functional"    # 功能正确性测试
    PERFORMANCE = "performance"  # 性能基准测试
```

### 任务类型的用途

✅ **可以用于:**
- 日志中的分类标记
- 统计报表 (例如: "今天运行了50个functional测试")
- 用户界面中的过滤
- 文档和工作流程的组织

❌ **不能用于:**
- 决定是否独占GPU
- 影响任务调度顺序
- 改变GPU资源分配
- 任何调度逻辑判断

### 示例

```python
# 调度器代码中完全不看task.task_type !
def _schedule_round(self):
    task = self.task_queue.pop_pending_task()
    
    # 只关心:
    # - task.gpu_id (用户指定的GPU)
    # - GPU的mode (exclusive/shared)
    # - GPU的资源限制 (max_tasks, memory)
    
    # 不关心:
    # - task.task_type ← 完全不看!
    
    available_gpu = self.gpu_manager.find_available_gpu(task.gpu_id)
    # ...
```

---

## 调度器设计

### 核心逻辑

```
每1秒钟运行一次调度轮次:

1. 从全局队列取出一个pending任务
2. 查找可用GPU (考虑GPU mode)
3. 如果找到可用GPU:
   → 分配任务到该GPU
   → 启动任务执行
4. 如果没有可用GPU:
   → 把任务放回队列
   → 下一轮再试
5. 重复步骤1-4,直到队列为空
```

### 代码实现

```python
# scheduler.py Line 35-56 (简化版)
def _schedule_round(self):
    """运行一轮调度"""
    while True:
        # 步骤1: 取任务
        task = self.task_queue.pop_pending_task()
        if task is None:
            break  # 没有待处理任务了
        
        # 步骤2: 找GPU
        gpu_id = self.gpu_manager.find_available_gpu(task.gpu_id)
        
        if gpu_id is None:
            # 步骤4: 没找到可用GPU,放回队列
            self.task_queue.global_queue.appendleft(task)
            break  # 本轮结束,1秒后再试
        
        # 步骤3: 分配并运行
        self.task_queue.assign_task_to_gpu(task.task_id, gpu_id)
        self.gpu_manager.add_task(gpu_id, task.task_id)
        
        # 在后台线程中运行任务
        threading.Thread(
            target=self.task_runner.run_task,
            args=(task, gpu_id),
            daemon=True
        ).start()
```

### GPU可用性检查

```python
# gpu_manager.py Line 209-246 (简化版)
def find_available_gpu(self, preferred_gpu=None):
    """找到一个可用的GPU"""
    
    # 如果用户指定了GPU
    if preferred_gpu is not None:
        gpu = self.gpus[preferred_gpu]
        if gpu.can_accept_task():  # ← 这里检查mode!
            return preferred_gpu
        return None  # 指定的GPU不可用
    
    # 找任意可用GPU
    for gpu_id, gpu in self.gpus.items():
        if gpu.can_accept_task():  # ← 这里检查mode!
            return gpu_id
    
    return None  # 没有可用GPU
```

### 核心判断逻辑

```python
# models.py Line 23-36
def can_accept_task(self) -> bool:
    """GPU能否接受新任务?"""
    
    # 检查1: GPU是否在线
    if self.status != GPUStatus.ONLINE:
        return False
    
    # 检查2: 如果是EXCLUSIVE模式
    if self.mode == GPUMode.EXCLUSIVE:
        return len(self.running_tasks) == 0  # 必须没有任务在运行
    
    # 检查3: SHARED模式 - 任务数限制
    if len(self.running_tasks) >= self.max_concurrent_tasks:
        return False
    
    # 检查4: SHARED模式 - 显存限制
    return self.current_memory_usage < self.memory_threshold
```

---

## 设计哲学

### 1. 关注点分离

```
GPU模式      ← 管理员控制 ← 资源管理层面
任务类型      ← 用户标注   ← 业务逻辑层面

两者独立,互不干扰
```

### 2. 显式控制

```
✓ 管理员明确设置GPU模式
✗ 系统不会自动改变模式

优点:
- 行为可预测
- 不会有意外
- 责任清晰
```

### 3. 优雅降级

```
模式切换时:
✓ 正在运行的任务继续执行
✗ 不中断、不强杀

优点:
- 数据完整性
- 不浪费计算
- 用户体验好
```

### 4. 简单调度

```
FIFO (先进先出)队列:
- 先提交的先执行
- 公平
- 不会饿死
- 易于理解和调试
```

---

## 常见问题 FAQ

### Q1: 我提交了一个performance任务,为什么它还是和其他任务一起运行?

**A:** 因为GPU当前是SHARED模式。任务类型不影响调度!

如果你希望独占GPU,需要:
```python
# 步骤1: 设置GPU为exclusive模式
client.set_gpu_mode(gpu_id=0, mode="exclusive")

# 步骤2: 提交任务
client.submit_task("benchmark.py", task_type="performance", gpu_id=0)
```

### Q2: GPU模式什么时候会自动切换?

**A:** 永远不会自动切换! 只能通过API手动切换。

### Q3: 我把GPU从shared切到exclusive,正在运行的3个任务会被杀掉吗?

**A:** 不会! 它们会继续运行直到完成。只有等这3个任务都完成后,新任务才会开始。

### Q4: 我能让performance任务优先于functional任务吗?

**A:** 当前版本不支持优先级。所有任务按提交顺序(FIFO)调度。

未来版本可能会增加优先级功能。

### Q5: 一个GPU可以同时处于exclusive和shared模式吗?

**A:** 不能。一个GPU在任意时刻只有一个模式。

### Q6: 如何知道当前GPU是什么模式?

**A:** 通过API查询:
```python
gpu_info = client.get_gpu(gpu_id=0)
print(f"Mode: {gpu_info['mode']}")
print(f"Running tasks: {gpu_info['running_task_count']}")
```

### Q7: shared模式下的max_concurrent_tasks怎么设置?

**A:** 根据你的任务特点:
- 轻量级测试: 可以设置5-10
- 显存密集型: 设置2-3
- 混合负载: 设置3-5 (默认)

```python
client.set_gpu_max_concurrent_tasks(gpu_id=0, max_tasks=5)
```

---

## 快速参考

### 我想要...怎么做?

| 需求 | 操作 |
|------|------|
| 让任务独占GPU | 1. `set_gpu_mode(id, "exclusive")` <br> 2. `submit_task(...)` |
| 让多个任务并行 | 1. `set_gpu_mode(id, "shared")` <br> 2. `submit_task(...)` (多次) |
| 限制并发数 | `set_gpu_max_concurrent_tasks(id, N)` |
| 标记任务用途 | `submit_task(..., task_type="functional")` |
| 指定使用某个GPU | `submit_task(..., gpu_id=0)` |
| 让任务自动选GPU | `submit_task(...)` (不指定gpu_id) |
| 查看GPU状态 | `get_gpu(id)` 或 `get_stats()` |
| 改变GPU行为 | `set_gpu_mode(id, mode)` |

### 代码快速查找

| 功能 | 文件 | 行数 |
|------|------|------|
| GPU模式定义 | `models.py` | 8-10 |
| 任务类型定义 | `models.py` | 13-15 |
| `can_accept_task()`核心逻辑 | `models.py` | 23-36 |
| 模式切换函数 | `gpu_manager.py` | 100-108 |
| GPU选择逻辑 | `gpu_manager.py` | 209-246 |
| 调度主循环 | `scheduler.py` | 28-33 |
| 调度一轮逻辑 | `scheduler.py` | 35-56 |
| 任务执行 | `task_runner.py` | 38-154 |

---

## 相关文档

- **[完整设计文档](DESIGN_ARCHITECTURE.md)** (English, 包含更详细的代码示例和场景分析)
- **[快速入门](QUICKSTART.md)** (如何使用server)
- **[完整文档](README.md)** (所有功能说明)
- **[GPU配置指南](GPU_CONFIG_GUIDE.md)** (如何配置GPU资源)
- **[示例代码](../examples/)** (实际代码示例)

---

## 文档版本

- **版本:** 1.0
- **日期:** 2025-10-08
- **语言:** 中文
- **状态:** 完整

有任何疑问,请参考完整的 [DESIGN_ARCHITECTURE.md](DESIGN_ARCHITECTURE.md) 文档。

