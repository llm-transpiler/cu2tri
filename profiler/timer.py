"""轻量级计时工具。

模块提供的 API 极少，仅包含上下文管理器、装饰器和一次性测量三个入口，
方便在现有脚本中插入性能监控。底层使用 ``time.monotonic`` 保证耗时不受
系统时间回拨影响，但这一细节对调用者透明。

典型用法::

    from profiler.timer import create_timer

    timer = create_timer()

    with timer.time("heavy_step"):
        run_heavy_step()

    @timer.wrap("critical_fn")
    def critical_fn(...):
        ...

    result = timer.measure("task", callable, *args, **kwargs)
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TimerSample:
    """Timing result for a single measurement."""

    label: str
    start: float
    end: float
    error: BaseException | None = None

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end - self.start

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration * 1000.0


class _TimerContext:
    """Internal context manager used by ``Timer``."""

    __slots__ = ("_label", "_report", "_start")

    def __init__(self, label: str, report: Callable[[TimerSample], None]):
        self._label = label
        self._report = report
        self._start: float | None = None

    def __enter__(self) -> "_TimerContext":
        self._start = monotonic()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool | None:
        start = self._start if self._start is not None else monotonic()
        end = monotonic()
        sample = TimerSample(self._label, start, end, exc)
        self._report(sample)
        # Propagate exceptions
        return False


class Timer:
    """Small helper providing context/decorator based timing."""

    def __init__(
        self,
        reporter: Callable[[TimerSample], None] | None = None,
        enabled: bool = True,
    ) -> None:
        self._reporter = reporter or _default_reporter
        self._enabled = enabled

    def time(self, label: str) -> _TimerContext:
        """Return a context manager that measures a code block."""
        if not self._enabled:
            return _NullContext(label)
        return _TimerContext(label, self._reporter)

    def wrap(self, label: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator version of ``time`` for quick instrumentation."""

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            if not self._enabled:
                return func

            def wrapped(*args: Any, **kwargs: Any) -> T:
                with self.time(label):
                    return func(*args, **kwargs)

            wrapped.__name__ = getattr(func, "__name__", "wrapped")
            wrapped.__doc__ = func.__doc__
            wrapped.__module__ = func.__module__
            return wrapped

        return decorator

    def measure(self, label: str, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Convenience helper to time a callable and return its result."""
        if not self._enabled:
            return func(*args, **kwargs)
        with self.time(label):
            return func(*args, **kwargs)

    def enable(self) -> None:
        """Enable timing."""
        self._enabled = True

    def disable(self) -> None:
        """Disable timing (context managers become no-ops)."""
        self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled


class _NullContext(_TimerContext):
    """No-op context used when the profiler is disabled."""

    def __init__(self, label: str) -> None:
        super().__init__(label, lambda sample: None)

    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool | None:
        return False


def create_timer(
    reporter: Callable[[TimerSample], None] | None = None,
    enabled: bool = True,
) -> Timer:
    """Factory for the default ``Timer``."""
    return Timer(reporter=reporter, enabled=enabled)


class HostTimer(Timer):
    """Timer 专用于宿主/CPU 侧计时，语义上与 ``Timer`` 一致。

    提供单独的类型别名，便于在需要区分 Host / Device 计时器的场景中
    进行类型标注或后续扩展。
    """

    pass


def create_host_timer(
    reporter: Callable[[TimerSample], None] | None = None,
    enabled: bool = True,
) -> HostTimer:
    """Factory for ``HostTimer``."""
    return HostTimer(reporter=reporter, enabled=enabled)


def _default_reporter(sample: TimerSample) -> None:
    """Default reporter prints a concise single-line summary."""
    status = "ERR" if sample.error else "OK"
    duration_ms = sample.duration_ms
    print(f"[timer] {sample.label}: {duration_ms:.3f} ms ({status})")


__all__ = [
    "Timer",
    "HostTimer",
    "TimerSample",
    "create_timer",
    "create_host_timer",
]
