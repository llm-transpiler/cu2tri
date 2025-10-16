"""多进程计时（集成 profiler.timer）示例。

每个子进程创建独立的计时器，通过自定义 reporter 收集样本（而非直接打印），
将样本通过 Queue 回传给主进程，由主进程统一汇总/打印，避免 I/O 抖动干扰测量。
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import List, Tuple

from profiler.timer import create_timer, TimerSample


@dataclass
class Segment:
    label: str
    duration_ms: float


def _worker(index: int, out_queue: mp.Queue) -> None:
    samples: List[Segment] = []

    def reporter(sample: TimerSample) -> None:
        # 收集样本，不在子进程内打印，避免 I/O 干扰
        samples.append(Segment(sample.label, sample.duration_ms))

    timer = create_timer(reporter=reporter)

    with timer.time(f"work.load.step1.{index}"):
        time.sleep(0.01 * (index + 1))

    with timer.time(f"work.load.step2.{index}"):
        time.sleep(0.02 * (index + 1))

    out_queue.put((index, samples))


def main() -> None:
    n_processes = 3
    out_queue: mp.Queue = mp.Queue()
    processes: List[mp.Process] = []

    for i in range(n_processes):
        p = mp.Process(target=_worker, args=(i, out_queue))
        p.start()
        processes.append(p)

    results: List[Tuple[int, List[Segment]]] = [out_queue.get() for _ in range(n_processes)]

    for p in processes:
        p.join()

    results.sort(key=lambda x: x[0])
    for index, segments in results:
        total = sum(s.duration_ms for s in segments)
        detail = ", ".join(f"{s.label}={s.duration_ms:.3f}ms" for s in segments)
        print(f"proc#{index} total={total:.3f} ms | {detail}")


if __name__ == "__main__":
    # 如需跨平台确定性，可显式：
    # import multiprocessing as mp; mp.set_start_method("spawn", force=True)
    main()


