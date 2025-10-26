"""Lightweight timing helpers for host-side instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic_ns, perf_counter_ns
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_NS_TO_MS = 1e-6


def ns_to_ms(ns: int | float) -> float:
    """Convert nanoseconds to milliseconds."""
    return float(ns) * _NS_TO_MS


def perf_counter_timestamp_ns() -> int:
    """Return high-resolution performance counter in nanoseconds."""
    return perf_counter_ns()


def perf_counter_timestamp_ms() -> float:
    """Return high-resolution performance counter in milliseconds."""
    return ns_to_ms(perf_counter_ns())


def monotonic_timestamp_ns() -> int:
    """Return monotonic clock value in nanoseconds."""
    return monotonic_ns()


def monotonic_timestamp_ms() -> float:
    """Return monotonic clock value in milliseconds."""
    return ns_to_ms(monotonic_ns())


def monotonic_elapsed_ms(start_ns: int) -> float:
    """Return elapsed milliseconds since the provided monotonic timestamp."""
    return ns_to_ms(monotonic_ns() - start_ns)


def perf_counter_elapsed_ms(start_ns: int) -> float:
    """Return elapsed milliseconds since the provided perf_counter timestamp."""
    return ns_to_ms(perf_counter_ns() - start_ns)


@dataclass(frozen=True)
class TimerSample:
    """Timing result for a single measurement."""

    label: str
    start_ns: int
    end_ns: int
    error: BaseException | None = None

    @property
    def start_ms(self) -> float:
        """Start timestamp in milliseconds."""
        return ns_to_ms(self.start_ns)

    @property
    def end_ms(self) -> float:
        """End timestamp in milliseconds."""
        return ns_to_ms(self.end_ns)

    @property
    def duration_ns(self) -> int:
        """Duration in nanoseconds."""
        return self.end_ns - self.start_ns

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return ns_to_ms(self.duration_ns)


class _TimerContext:
    """Internal context manager used by Timer."""

    __slots__ = ("_label", "_report", "_clock_ns", "_start")

    def __init__(
        self,
        label: str,
        report: Callable[[TimerSample], None],
        clock_ns: Callable[[], int],
    ) -> None:
        self._label = label
        self._report = report
        self._clock_ns = clock_ns
        self._start: int | None = None

    def __enter__(self) -> "_TimerContext":
        self._start = self._clock_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool | None:
        start = self._start if self._start is not None else self._clock_ns()
        end = self._clock_ns()
        sample = TimerSample(self._label, start_ns=start, end_ns=end, error=exc)
        self._report(sample)
        return False


class Timer:
    """Small helper providing context/decorator based timing."""

    def __init__(
        self,
        reporter: Callable[[TimerSample], None] | None = None,
        enabled: bool = True,
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        self._reporter = reporter or _default_reporter
        self._enabled = enabled
        self._clock_ns = clock_ns or perf_counter_ns

    def time(self, label: str) -> _TimerContext:
        """Return a context manager that measures a code block."""
        if not self._enabled:
            return _NullContext(label)
        return _TimerContext(label, self._reporter, self._clock_ns)

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
        """Disable timing (contexts become no-ops)."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Return whether the timer is enabled."""
        return self._enabled


class _NullContext(_TimerContext):
    """No-op context used when the profiler is disabled."""

    def __init__(self, label: str) -> None:
        super().__init__(label, lambda sample: None, lambda: 0)

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
    clock_ns: Callable[[], int] | None = None,
) -> Timer:
    """Factory for the default Timer."""
    return Timer(reporter=reporter, enabled=enabled, clock_ns=clock_ns)


class HostTimer(Timer):
    """Timer specialized for host/CPU measurements."""

    pass


def create_host_timer(
    reporter: Callable[[TimerSample], None] | None = None,
    enabled: bool = True,
    clock_ns: Callable[[], int] | None = None,
) -> HostTimer:
    """Factory for HostTimer."""
    return HostTimer(reporter=reporter, enabled=enabled, clock_ns=clock_ns)


def _default_reporter(sample: TimerSample) -> None:
    """Default reporter prints a concise single-line summary."""
    status = "ERR" if sample.error else "OK"
    print(f"[timer] {sample.label}: {sample.duration_ms:.3f} ms ({status})")


__all__ = [
    "Timer",
    "HostTimer",
    "TimerSample",
    "create_timer",
    "create_host_timer",
    "ns_to_ms",
    "perf_counter_timestamp_ns",
    "perf_counter_timestamp_ms",
    "perf_counter_elapsed_ms",
    "monotonic_timestamp_ns",
    "monotonic_timestamp_ms",
    "monotonic_elapsed_ms",
]
