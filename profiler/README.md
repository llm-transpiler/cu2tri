# 单调时间辅助工具

`profiler/timer.py` 提供了一套基于 `time.monotonic()` 的轻量级计时工具，方便在现有代码中快速插入稳定的性能测量，而无需引入复杂的分析框架。

## 设计目标
- **使用单调时钟：** `time.monotonic()` 不会受 NTP 或手动校时影响，时长计算稳定可靠。仍可同时保留 `datetime.now()` 等绝对时间，用于日志或业务记录。
- **轻量易插拔：** API 只有上下文管理器、装饰器和一次性测量三个入口，插入代码改动极小。
- **可自定义 Reporter：** 所有测量结果都会包装成 `TimerSample` 并交给 reporter 回调。默认 reporter 输出一行摘要；也可替换为日志、指标或其他收集器。
- **运行时开关：** 可随时启用/禁用，便于在性能验证结束后关闭额外开销，但保留代码入口。

## 计时来源与范围
- 本工具使用 Python 进程内的单调时钟（CPU/系统层时间），适合衡量“从调用到返回”这类整体耗时。
- 如果包裹的是 GPU/NPU 操作，记录的仍是主机侧发起调用到任务完成的时间，而非 GPU 内核的精确执行时长。
- 针对 GPU、NPU 的更细粒度 profiling，可在此基础上引入专用工具，然后把关键阶段的 host 级总耗时通过本工具汇总，做到互补。

## 设备专用封装（Host 级）
`profiler/device_profilers.py` 提供了带有设备前缀的封装，便于区分不同后端：
- 这些封装继承自 `HostTimer`，用于在 Host 侧衡量与各类设备交互的耗时。
- `nvgpu_timer()`：NVIDIA GPU（NVGPU）宿主侧耗时；
- `amdgpu_timer()`：AMD GPU 宿主侧耗时；
- `bang_timer()`：寒武纪 BANG/NPU 宿主侧耗时；
- `create_device_timer(device_name)`：自定义设备名称。

示例：
```python
from profiler.device_profilers import nvgpu_timer

timer = nvgpu_timer()

with timer.time("graph_build"):
    build_cuda_graph()
# 输出类似：[timer] NVGPU:graph_build: 5.812 ms (OK)
```

## 核心类型
- `Timer`：基础计时器，实现上下文、装饰器、一次性测量等能力。
- `HostTimer`：继承自 `Timer`，语义上限定为宿主/CPU 侧计时，方便在 Host / Device 混用场景中做类型区分。
- `TimerSample`：不可变数据结构，包含 `label`、`start`、`end`、`duration`、`duration_ms` 和 `error`。
- `create_timer()` / `create_host_timer()`：便捷工厂函数，分别返回 `Timer` 与 `HostTimer`。

## 快速上手
```python
from profiler.timer import create_host_timer

timer = create_host_timer()

with timer.time("warmup"):
    warm_up_gpu()

@timer.wrap("batch_inference")
def run_batch(inputs):
    return engine.run(inputs)

result = timer.measure("export", export_model, path="model.onnx")
```

默认 reporter 的输出示例：
```
[timer] warmup: 12.415 ms (OK)
[timer] batch_inference: 7.202 ms (OK)
[timer] export: 31.884 ms (OK)
```

更多完整示例可参考 `profiler/examples/`：
- `basic_usage.py`：上下文、装饰器、一次性测量的综合演示。
- `custom_reporter.py`：自定义 reporter，将结果写到 logging。
- `asyncio_usage.py`：在 `asyncio` 协程内的计时示例。
- `multiprocessing_usage.py`：子进程独立计时示例。
- `device_usage.py`：NVGPU/AMDGPU/BANG 封装示例。

## 自定义 Reporter
使用自定义 reporter 输出结构化日志或写入指标系统：
```python
import logging
from profiler.timer import create_host_timer, TimerSample

logger = logging.getLogger("perf")

def reporter(sample: TimerSample) -> None:
    logger.info(
        "label=%s duration_ms=%.3f status=%s",
        sample.label,
        sample.duration_ms,
        "error" if sample.error else "ok",
    )

timer = create_host_timer(reporter=reporter)
```

`time`、`wrap`、`measure` 三个入口都会调用 reporter，并传入相同的 `TimerSample`，方便对接 Prometheus、OpenTelemetry 或内部统计流程。

## 运行时控制
- `timer.disable()`：切换为无操作上下文，关闭额外开销。
- `timer.enable()`：重新启用计时，无需修改调用代码。
- `timer.is_enabled()`：查看当前开关状态，便于按需切换。

## 异常处理
- 计时区域内部抛出的异常会照常传播。
- `TimerSample.error` 会记录对应的异常对象，方便 reporter 标记失败。

## 实现说明
- `_TimerContext` 是内部上下文管理器，用于记录开始和结束的单调时间；`_NullContext` 则是禁用时的空实现。
- 模块没有全局状态或异步钩子，适合在同步或多线程场景中直接使用。`time.monotonic()` 是进程范围的，线程安全，无需额外锁。

## 进阶用法建议
- 使用有意义的 label，如 `"load_batch"`、`"preprocess"`、`"inference"`、`"postprocess"`，便于后续分析。
- 在服务中同时保留绝对时间日志：用 `Timer` 记录耗时，继续使用 `datetime` 系列记录业务事件时间点。
- 与现有数据结构结合（例如 NVGPU 的 `Task`）：可以将 `TimerSample.duration_ms` 写入任务记录，以补充 wall-clock 信息。

更多细节可参考 `profiler/timer.py` 中的注释与实现。
