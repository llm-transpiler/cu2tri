"""使用 Timer 的多进程示例。

每个子进程都会创建自己的 profiler 并记录任务耗时。
"""

from __future__ import annotations

import multiprocessing as mp
import time

from profiler.timer import create_timer


def worker_task(idx: int) -> None:
    timer = create_timer()
    with timer.time(f"worker-{idx}"):
        time.sleep(0.01 * (idx + 1))


def main() -> None:
    processes: list[mp.Process] = []
    for i in range(3):
        p = mp.Process(target=worker_task, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/profiler/examples/multiprocessing_usage.py
[timer] worker-1: 34.196 ms (OK)
[timer] worker-2: 30.857 ms (OK)
[timer] worker-0: 23.469 ms (OK)
这是多进程计时的正常现象：每个子进程各自计时 sleep 的区间，但实际耗时会包含调度与唤醒延迟，打印顺序也不保证与 idx 一致，因此看到 23/30/34ms 且顺序无序是预期的。
为什么比 10/20/30ms 大: time.sleep 是“至少睡眠”语义，实际会多出内核调度/唤醒抖动；再加上进入/退出上下文与打印的微小开销。
为什么顺序无序: 三个子进程并行执行，stdout 合流到父进程，谁先被调度完成就先打印，和 idx 无关。
“spawn” 的影响: 你强制使用 spawn（Linux 默认是 fork）。spawn 会为每个子进程启动全新解释器并重新导入模块，系统负载更高，调度抖动更明显（虽然启动开销不在计时区间内，但会影响整体调度）。
'''

'''
将sleep倍数调到1000，可以看到计时明显变大，并且顺序也有变化
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/profiler/examples/multiprocessing_usage.py
[timer] worker-0: 1004.353 ms (OK)
[timer] worker-1: 2008.868 ms (OK)
[timer] worker-2: 3000.071 ms (OK)
'''