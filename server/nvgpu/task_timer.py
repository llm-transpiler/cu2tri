"""Utilities for measuring task lifecycle phases with host timers."""

from __future__ import annotations


from profiler.timer import HostTimer, TimerSample, create_host_timer


class TaskTimer:
    """Track task lifecycle using HostTimer contexts.

    The tracker keeps lightweight open contexts that can be started and stopped
    across different modules (queue, scheduler, runner) without forcing them
    into a single scope.
    """

    def __init__(self, task_id: str | None = None) -> None:
        self._samples: list[TimerSample] = []
        self._durations_ms: dict[str, float] = {}
        self._active_contexts: dict[str, _ManagedContext] = {}
        self._timer: HostTimer = create_host_timer(
            reporter=self._capture_sample,
            enabled=True,
        )
        self._task_id = task_id

    def _capture_sample(self, sample: TimerSample) -> None:
        """Capture timer sample, storing in milliseconds."""
        self._samples.append(sample)
        # Accumulate durations in milliseconds for repeated segments.
        previous = self._durations_ms.get(sample.label, 0.0)
        self._durations_ms[sample.label] = previous + sample.duration_ms

    def start(self, label: str) -> None:
        """Start timing segment if not already active."""
        if label in self._active_contexts:
            return
        ctx = _ManagedContext(self._timer.time(label))
        ctx.enter()
        self._active_contexts[label] = ctx

    def stop(self, label: str) -> float | None:
        """Stop timing segment and return duration in milliseconds."""
        ctx = self._active_contexts.pop(label, None)
        if not ctx:
            return None
        ctx.exit()
        return self._durations_ms.get(label)

    def get_duration_ms(self, label: str) -> float | None:
        """Get measured duration in milliseconds."""
        return self._durations_ms.get(label)

    def as_dict(self) -> dict[str, float]:
        """Return a copy of collected durations."""
        return dict(self._durations_ms)

    def reset(self) -> None:
        """Reset collected samples and contexts."""
        for ctx in list(self._active_contexts):
            self.stop(ctx)
        self._samples.clear()
        self._durations_ms.clear()


class _ManagedContext:
    """Helper to manually manage timer context enter/exit."""

    def __init__(self, ctx) -> None:  # ctx typing: _TimerContext
        self._ctx = ctx
        self._entered = False

    def enter(self) -> None:
        if not self._entered:
            self._ctx.__enter__()
            self._entered = True

    def exit(self) -> None:
        if self._entered:
            self._ctx.__exit__(None, None, None)
            self._entered = False
