## NVGPU 性能测试说明

本目录下是 cu2tri 流水线相关的 **性能测试文档与辅助代码**。  
性能测试目前是 **仅支持 NVGPU** 的，并且已经集成进现有的 `cu2til.llm_trans` 服务中，**不再使用独立的 `run_perf_tests.py` 脚本**。

整体上有两类 perf 流程：

- **按 case 的单次 perf**：在正确性测试通过后，由 `services` 自动触发。
- **基于配置的批量 perf**：根据 `config/perf_runs.yaml` 中列出的历史 run 进行性能回放。

> 默认行为：  
> - CLI 参数中 **不显式启用 perf**（未传 `--enable-perf`）时，**不会跑任何 perf**；  
> - 只有当你传入 `--enable-perf` 且 NVGPU 可用时，才会对通过正确性测试的 case 额外执行 perf；  
> - 本文档只说明 **如何在需要时启用 perf**，默认仍是关闭状态。

---

## 1. 按 case 的 NVGPU 性能测试（集成在主流程中）

当你运行主流程（CUDA→Triton 或 Triton→CUTE）并启用 NVGPU 时，每个 **成功通过正确性测试的 case** 可以额外跑一轮性能测试。

### 1.1 主流程入口（示例）

```bash
python -m cu2til.llm_trans \
  --direction cu2tri \
  --testset xpiler \
  --model <MODEL_NAME> \
  --use-nvgpu \
  --nvgpu-server http://localhost:8080
```

- **不会自动跑 perf**，除非同时满足：
  - `--use-nvgpu` 开启，且 NVGPU 可用；
  - CLI 中显式传入 `--enable-perf`（对应 `args.enable_perf` 为 True）。
- 在 `services/testing.py` 中，当某个 case 的某一轮测试 **首次通过** 时，会尝试调用：
  - `cu2til.llm_trans.services.perf.run_perf_nvgpu` 进行 NVGPU 性能测试；
  - 然后调用 `cu2til.llm_trans.services.perf.record_perf_result` 记录结果。

### 1.2 单 case perf 的执行与日志

- **执行脚本**：
  - `direction=cu2tri`：在对应的 `attempt_xx` 目录下执行 `check_triton*.py`；
  - `direction=tri2cute`：执行 `check_cute.py`。
- **日志位置**（以某个 case 的 `attempt_01` 为例）：
  - `.../attempt_01/logs/perf/perf.log`  
    - 保存 NVGPU 任务日志（stdout/stderr + 服务器计时信息）。
  - `.../attempt_01/perf.json`  
    - 如果脚本输出了结构化 perf JSON，会被解析并写入此文件。
- **聚合记录**：
  - `record_perf_result` 会把本次 perf 的：
    - 内核性能数据（来自 `perf.json` 或 stdout）；
    - NVGPU 服务端时间信息（pending/queue/waiting/running/total 等）  
    一并写入主 JSONL 日志（方便后续统计和可视化）。

通常你**不需要**在用户代码中直接调用 `run_perf_nvgpu`，它已经挂在 `services.testing.run_testing_loop` 的成功路径里，只要在 CLI 或内部构造 `context.args` 时设置 `enable_perf=True` 即可。

---

## 2. 基于配置的批量 perf 回放（主要用于 xpiler）

如果你已经完成了大量 cu2tri 转译与测试，并且在 `runs/` 与 `stats/` 下积累了历史结果，可以通过一个统一的配置文件来 **批量重跑性能测试**。

### 2.1 配置文件：`config/perf_runs.yaml`

- 配置结构为：

```yaml
cu2tri:
  xpiler:
    gpt_oss_20b:
      - 20251125_082610
      - 20251202_030418
    gpt_oss_120b:
      - 20251126_173846
      - 20251202_030406
    ...
```

- 语义：
  - **第一层 key**：`direction`（如 `cu2tri`）；
  - **第二层 key**：`testset`（如 `xpiler`）；
  - **第三层 key**：`model_tag`（与 `runs/<direction>/<testset>/<model_tag>/` 对应的目录名）；
  - **列表值**：要做 perf 回放的 `run_timestamp`（即对应的时间戳子目录名）。

只要在这里维护好哪些 `(direction, testset, model_tag, timestamp)` 需要做性能回放，批量 perf 服务就会自动在对应的 `runs/` 与 `stats/` 路径下查找数据并提交 NVGPU 任务。

### 2.2 执行批量 perf：命令行与 Python 接口

#### 2.2.1 通过命令行一次性跑完 `perf_runs.yaml` 中配置的所有 run

在项目根目录下，可以直接使用：

```bash
python -m cu2til.llm_trans.perf \
  --direction cu2tri \
  --testset xpiler \
  --model gpt_oss_20b \
  --nvgpu-server http://localhost:8080
```

- 该命令会：
  - 使用与主 CLI 相同的配置系统构造一个 `RuntimeContext`；
  - 读取 `config/perf_runs.yaml` 中 `cu2tri/xpiler` 下的所有 `(model_tag, timestamp)`；
  - 对每个组合执行下文所述的批量 perf 逻辑；
  - 最后在标准输出打印类似：

    ```json
    {
      "success": true,
      "attempts": 42,
      "successful_attempts": 42,
      "runs": 5
    }
    ```

#### 2.2.2 从 Python 代码中调用：`services.perf.run`

- 批量 perf 的核心服务入口是：
  - `cu2til.llm_trans.services.perf.run(context)`
- 它会：
  - 读取当前 `context.settings.direction` 与 `context.args.testset`；  
    - 对 xpiler 情况：`direction=cu2tri`、`testset=xpiler`；
  - 从 `config/perf_runs.yaml` 中找到对应的 `(model_tag, timestamp)` 列表；
  - 对每一个组合：
    - 读取：
      - `runs/<direction>/<testset>/<model_tag>/<timestamp>/...`
      - `stats/<direction>/<testset>/<model_tag>/<timestamp>/case_success.json`
    - 从 `case_success.json` 中找到所有成功的 attempt；
    - 对每个成功的 attempt，在其 `attempt_xx` 目录下执行 `check_triton_gpu_all.py` 的 NVGPU 性能任务；
    - 将 NVGPU 任务日志写入 `attempt_xx/logs/perf/perf.log`。

> 注意：批量 perf 只依赖已有的 `runs/` 与 `stats/`，不会重新跑 LLM 翻译或 correctness，**只做性能回放**。

### 2.3 从 Python 调用批量 perf 的完整示例

```python
from cu2til.llm_trans.cli import prepare_context
from cu2til.llm_trans.services.perf import run as run_perf_service
import asyncio

# 构造一个 RuntimeContext，direction/testset 与 perf_runs.yaml 对应，
# model 只需是一个合法模型名即可（perf 本身不再使用 LLM）。
ctx = prepare_context([
    "--direction", "cu2tri",
    "--testset", "xpiler",
    "--model", "gpt_oss_20b",           # 任意已注册模型即可
    "--use-nvgpu",
    "--nvgpu-server", "http://localhost:8080",
    "--enable-perf",
])

summary = asyncio.run(run_perf_service(ctx))
print(summary)
```

返回结果形如：

```python
{
    "success": True,
    "attempts": 42,
    "successful_attempts": 42,
    "runs": 5,
}
```

---

## 3. 如何扩展 / 调整 perf 配置

- **步骤 1：先跑正常流水线**  
  - 确保存在：
    - `runs/<direction>/<testset>/<model_tag>/<timestamp>/...`
    - `stats/<direction>/<testset>/<model_tag>/<timestamp>/case_success.json`
  - 其中 `case_success.json` 里包含了成功 attempt 的信息。

- **步骤 2：编辑 `config/perf_runs.yaml`**  
  - 在对应的 `direction` / `testset` / `model_tag` 下，追加新的 `timestamp`。
  - 一般可以只挑最近若干次 run 做 perf。

- **步骤 3：调用 `services.perf.run` 批量跑 perf**  
  - 构造 `context` 时 direction/testset 与 `perf_runs.yaml` 对齐；
  - 确保 `--use-nvgpu` 和 `--nvgpu-server` 指向可用的 NVGPU 服务；
  - 调用 `run_perf_service(context)`，查看返回的 summary 与各 `attempt_xx/logs/perf/perf.log`。

整个设计中：  
- **默认 CLI 行为仍然是 “不跑 perf”**（`--no-perf` 默认 True）；  
- 只有在你显式关闭 `no_perf` 或在批量 perf 中调用 `services.perf.run` 时，才会真正发起 NVGPU 性能测试。  
这保证了主流程的默认开销不变，同时为需要做性能分析的场景提供了一套统一、可配置的入口。


