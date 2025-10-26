from __future__ import annotations


class ManagedContext:
    """Helper to manually manage a context manager's enter/exit lifecycle.

    This wraps an existing context manager, allowing start/stop style control
    across scopes without leaking the internal context type.
    """

    def __init__(self, ctx) -> None:  # ctx: any context manager with __enter__/__exit__
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


__all__ = ["ManagedContext"]


