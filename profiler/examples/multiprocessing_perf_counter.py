"""多进程计时（perf_counter_ns 直测）示例。

每个子进程在自身进程内用 `time.perf_counter_ns()` 记录目标工作负载的开始/结束时间，
通过 `multiprocessing.Queue` 将结构化结果回传到主进程，主进程统一排序与打印。
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Dict, List


def now_ns() -> int:
    return time.perf_counter_ns()


def _worker(index: int, base_sleep_seconds: float, out_queue: mp.Queue) -> None:
    """在子进程内测量目标工作区间，避免 I/O 干扰计时。"""
    start_ns = now_ns()
    print(f"start_ns: {start_ns}")
    try:
        # 真实工作负载（示例使用 sleep）：不同 index 给予不同耗时
        time.sleep(base_sleep_seconds * (index + 1))
    finally:
        end_ns = now_ns()
    print(f"end_ns: {end_ns}")
    duration_ms = (end_ns - start_ns) / 1e6
    out_queue.put(
        {
            "index": index,
            "pid": os.getpid(),
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ms": duration_ms,
        }
    )


def main() -> None:
    n_processes = 3
    base_sleep_seconds = 0.01  # 放大可降低相对误差，如 0.05 或 0.1

    out_queue: mp.Queue = mp.Queue()
    processes: List[mp.Process] = []

    for i in range(n_processes):
        p = mp.Process(target=_worker, args=(i, base_sleep_seconds, out_queue))
        p.start()
        processes.append(p)

    results: List[Dict] = [out_queue.get() for _ in range(n_processes)]

    for p in processes:
        p.join()

    # 统一排序与打印，避免子进程内频繁 I/O 干扰计时
    results.sort(key=lambda r: r["index"])
    for r in results:
        print(
            f"proc#{r['index']} pid={r['pid']} duration={r['duration_ms']:.3f} ms"
        )


if __name__ == "__main__":
    # 如需在 Linux 降低调度抖动，可考虑：
    # import multiprocessing as mp; mp.set_start_method("fork", force=True)
    main()


