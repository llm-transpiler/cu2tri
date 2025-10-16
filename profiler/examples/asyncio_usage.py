"""使用 Timer 的 asyncio 示例。

演示在异步任务中使用同步上下文管理器与装饰器。
"""

from __future__ import annotations

import asyncio

from profiler.timer import create_timer


timer = create_timer()


@timer.wrap("async_task")
async def async_task(duration: float) -> str:
    await asyncio.sleep(duration)
    return f"slept {duration:.3f}s"


async def main() -> None:
    with timer.time("gather"):
        results = await asyncio.gather(
            async_task(0.01),
            async_task(0.02),
            async_task(0.03),
        )

    for item in results:
        print("task result:", item)


if __name__ == "__main__":
    asyncio.run(main())
