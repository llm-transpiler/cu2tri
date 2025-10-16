## 多进程下如何获得尽可能准确的时间计时

本指南总结在 Python 多进程（multiprocessing）环境中进行“尽可能准确”的用时测量的方法与注意事项，并提供一份可复制的参考实现。

### 核心结论
- 使用单调时钟：优先 `time.perf_counter_ns()`（纳秒精度，单调递增）。
- 将测量范围最小化：只包裹真实工作负载，不要把进程启动/通信/日志打印包含在计时区间内。
- 每个子进程独立计时：在子进程内获取开始/结束时间，再把结果回传到主进程汇总。
- 控制启动方式：在 Linux 环境下可用 `fork` 降低调度抖动；跨平台或对可重入性敏感时使用 `spawn` 并放大负载以降低相对误差。
- 减少干扰：关闭/延后日志，使用无锁/批量 I/O，将 CPU 亲和性与进程数与核心数匹配，降低频繁上下文切换。

### 推荐实现（一）：轻量可靠（基于 perf_counter_ns 与最小通信）

```python
"""多进程计时：子进程内使用 perf_counter_ns，主进程汇总。

适合：CPU 计算或 I/O 任务的相对/绝对耗时统计，追求工程可用与低侵入。
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Callable


def now_ns() -> int:
    return time.perf_counter_ns()


@dataclass
class MeasureResult:
    index: int
    pid: int
    start_ns: int
    end_ns: int
    duration_ms: float


def measure_callable(index: int, fn: Callable[[], None]) -> MeasureResult:
    start = now_ns()
    try:
        fn()
    finally:
        end = now_ns()
    return MeasureResult(
        index=index,
        pid=os.getpid(),
        start_ns=start,
        end_ns=end,
        duration_ms=(end - start) / 1e6,
    )


def _worker(index: int, task_args: tuple, task_kwargs: dict, out_queue: mp.Queue) -> None:
    # 仅在子进程内进行计时与计算，避免 stdout 干扰计时
    def workload() -> None:
        # 将真实负载写在这里；示例为 sleep
        time.sleep(*task_args, **task_kwargs)

    result = measure_callable(index, workload)
    out_queue.put(result)


def run_processes(n: int = 4, sleep_seconds: float = 0.05, start_method: str | None = None) -> list[MeasureResult]:
    if start_method:
        mp.set_start_method(start_method, force=True)

    out_queue: mp.Queue = mp.Queue()
    processes: list[mp.Process] = []
    for i in range(n):
        p = mp.Process(target=_worker, args=(i, (sleep_seconds * (i + 1),), {}, out_queue))
        p.start()
        processes.append(p)

    results: list[MeasureResult] = []
    for _ in range(n):
        results.append(out_queue.get())

    for p in processes:
        p.join()

    # 主进程统一排序与打印，避免在子进程打印造成干扰
    results.sort(key=lambda r: r.index)
    for r in results:
        print(f"proc#{r.index} pid={r.pid} duration={r.duration_ms:.3f} ms")

    return results


if __name__ == "__main__":
    # Linux 下如需更低抖动：start_method="fork"；跨平台或需要确定性：start_method="spawn"
    run_processes(n=3, sleep_seconds=0.03, start_method=None)
```

要点：
- 采用 `Queue` 单次回传结构化结果，不在子进程里频繁打印。
- 由主进程统一排序和输出，保证可读性和可比性。
- 将 `perf_counter_ns` 换算为毫秒打印，同时保留原始纳秒时间便于诊断。

### 推荐实现（二）：结合现有 profiler.timer（可插桩、统计分段）

当你需要统一的计时 API（上下文、装饰器、分段统计）并且希望日志格式一致，可以在子进程内创建独立计时器，将原始样本（label、duration_ms）汇总回主进程，不直接在子进程打印。

```python
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any

from profiler.timer import create_timer, TimerSample


@dataclass
class Segment:
    label: str
    duration_ms: float


def _worker_with_profiler(index: int, out_queue: mp.Queue) -> None:
    samples: list[Segment] = []

    def reporter(sample: TimerSample) -> None:
        # 将样本收集到列表中，避免子进程直接日志 I/O 干扰
        samples.append(Segment(sample.label, sample.duration_ms))

    timer = create_timer(reporter=reporter)
    with timer.time(f"stage.sleep.{index}"):
        time.sleep(0.05 * (index + 1))

    out_queue.put((index, samples))


def run_with_profiler(n: int = 3) -> list[tuple[int, list[Segment]]]:
    out_queue: mp.Queue = mp.Queue()
    processes: list[mp.Process] = []
    for i in range(n):
        p = mp.Process(target=_worker_with_profiler, args=(i, out_queue))
        p.start()
        processes.append(p)

    results: list[tuple[int, list[Segment]]] = []
    for _ in range(n):
        results.append(out_queue.get())

    for p in processes:
        p.join()

    results.sort(key=lambda x: x[0])
    for index, segments in results:
        total = sum(s.duration_ms for s in segments)
        print(f"proc#{index} total={total:.3f} ms segments={[ (s.label, round(s.duration_ms,2)) for s in segments ]}")

    return results


if __name__ == "__main__":
    run_with_profiler()
```

要点：
- 使用 `reporter` 收集样本，主进程统一打印，避免 I/O 抖动。
- 每个子进程的计时器互不影响，可独立插桩多个阶段。

### 精度与稳定性建议
- 放大工作负载：把目标操作时长提升到 50ms+，相对误差显著下降。
- 控制并发度：进程数不要超过物理核心数；必要时设置 CPU 亲和性（taskset/psutil）。
- 降低系统噪声：减少磁盘 I/O、网络、日志；避免高负载背景任务。
- 统一时钟源：在同一个进程内比较相对时间用 `perf_counter_ns`；跨进程对齐仅做相对比较，避免混用 `time.time()`。
- 进程启动方式：
  - Linux：`fork` 延迟更低、抖动更小；
  - 跨平台/库安全：`spawn` 更安全但调度抖动更显著。
- 统计与可重复性：多次运行取中位数与分位数（p50/p90/p99），不要依赖单次值。

### 常见误区
- 在子进程里边计时边打印，导致 I/O 与锁竞争放大抖动。
- 混合使用不同时钟（`time.time()` 与 `perf_counter()`）做差，得到不可靠结果。
- 把进程启动、队列阻塞、序列化开销一并算入目标操作的测量区间。

---

如需将上述方案对接到现有 `NVGPU` / `TaskRunner` 日志体系，建议：
1) 在子进程侧仅收集样本；2) 通过 IPC 传回；3) 由主进程以统一 logger 打印或写文件，确保时序与格式一致。


