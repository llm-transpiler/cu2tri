from __future__ import annotations


def format_ms(value: float | None, ms_format: str) -> str:
    if value is None:
        return "N/A"
    fmt = "{:,.3f}" if ms_format == "comma" else "{:.3f}"
    return f"{fmt.format(value)} ms"


__all__ = ["format_ms"]
