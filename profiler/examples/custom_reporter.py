"""自定义 reporter 示例。

展示如何将测量结果发送到 logging 模块。
"""

from __future__ import annotations

import logging
import time

from profiler.timer import TimerSample, create_timer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("perf")


def reporter(sample: TimerSample) -> None:
    status = "error" if sample.error else "ok"
    logger.info(
        "label=%s duration_ms=%.3f status=%s",
        sample.label,
        sample.duration_ms,
        status,
    )


timer = create_timer(reporter=reporter)


def work(duration: float) -> None:
    time.sleep(duration)


def main() -> None:
    with timer.time("prepare"):
        work(0.015)

    try:
        with timer.time("may_fail"):
            raise RuntimeError("demo error")
    except RuntimeError:
        pass


if __name__ == "__main__":
    main()
