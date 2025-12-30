#!/usr/bin/env python3
"""
GPU monitor log analyzer

Features:
- Auto-detect common log formats (JSON lines, key=value, CSV including nvidia-smi --query outputs)
- Parse timestamp, gpu index, utilization, optional pid/process count, optional memory
- Output a tidy CSV with columns: timestamp,gpu_index,utilization,pid_count,memory_used_mb
- Generate charts (PNG and/or HTML) showing per-GPU utilization over time and task concurrency

Usage:
  python tools/gpu_log_analyzer.py \
    --input /data/apps/project/cu2tri/logs/monitor.log \
    --output-dir /data/apps/project/cu2tri/logs/monitor_plots \
    --html  # also write interactive HTML via plotly if available

Notes:
- The parser is heuristic; if auto-detection fails, you can hint with --format json|kv|csv
- If your log contains a process list per GPU (e.g., processes:[pid,...]), pid_count will be extracted
- If only utilization exists, charts for task counts will be skipped gracefully
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


def _parse_timestamp(value: str) -> Optional[dt.datetime]:
    """Try to parse a timestamp from a variety of common formats.

    Supported examples:
    - 2025-10-21 12:34:56.123456
    - 2025-10-21T12:34:56.123Z
    - 2025/10/21 12:34:56
    - 12:34:56.123 (today's date assumed)
    """
    value = value.strip()
    patterns = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%H:%M:%S.%f",
        "%H:%M:%S",
    ]
    for fmt in patterns:
        try:
            ts = dt.datetime.strptime(value, fmt)
            if ts.year == 1900:  # time-only
                today = dt.date.today()
                ts = dt.datetime.combine(today, ts.time())
            return ts
        except ValueError:
            continue
    # ISO 8601 with timezone offset like 2025-10-21T07:12:45+08:00
    try:
        return dt.datetime.fromisoformat(value)
    except Exception:
        pass
    return None


def _extract_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def _extract_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


class LineParser:
    """Heuristic line parser supporting JSON, key=value, and CSV-like logs."""

    # key normalization map
    KEY_ALIASES = {
        "ts": "timestamp",
        "time": "timestamp",
        "datetime": "timestamp",
        "date": "timestamp",
        "gpu": "gpu_index",
        "gpu_id": "gpu_index",
        "index": "gpu_index",
        "id": "gpu_index",
        "util": "utilization",
        "utilization.gpu": "utilization",
        "gpu_util": "utilization",
        "gpu_percent": "utilization",
        # memory/task metrics in monitor.log
        "current_memory_usage": "memory_percent",  # fraction 0..1 or percent string
        "pid_count": "task_count",
        "procs": "task_count",
        "running_task_count": "task_count",
        "processes": "processes",
        "running_tasks": "processes",
        "mem": "memory_used_mb",
        "memory.used": "memory_used_mb",
        "memory_used": "memory_used_mb",
        "mem_used": "memory_used_mb",
    }

    KV_PATTERN = re.compile(r"(?:\b|,|\s)([A-Za-z0-9_.]+)\s*=\s*([^,\s]+)")
    GPU_UTIL_PATTERN = re.compile(r"GPU\s*(\d+)\D+?(\d{1,3})%")

    def __init__(self, forced_format: Optional[str] = None) -> None:
        if forced_format not in {None, "json", "kv", "csv"}:
            raise ValueError("--format must be one of: json, kv, csv")
        self.forced_format = forced_format
        self.header_keys: Optional[List[str]] = None

    def parse_line(self, line: str) -> List[Dict[str, object]]:
        line = line.strip()
        if not line:
            return []

        # Try forced format first
        strategies = (
            [self._parse_json] if self.forced_format == "json" else
            [self._parse_kv] if self.forced_format == "kv" else
            [self._parse_csv] if self.forced_format == "csv" else
            [self._parse_ts_plus_json, self._parse_json, self._parse_kv, self._parse_csv, self._parse_fallback_gpu_block]
        )

        for strat in strategies:
            try:
                rows = strat(line)
                if rows:
                    return rows
            except Exception:
                # proceed to next strategy
                pass
        return []

    def _normalize_record(self, rec: Dict[str, object]) -> Optional[Dict[str, object]]:
        norm: Dict[str, object] = {}
        for k, v in rec.items():
            key = self.KEY_ALIASES.get(k, k)
            norm[key] = v

        # timestamp
        ts = norm.get("timestamp")
        if isinstance(ts, str):
            parsed = _parse_timestamp(ts)
            if parsed is None:
                # allow numeric epoch
                try:
                    parsed = dt.datetime.fromtimestamp(float(ts))
                except Exception:
                    parsed = None
            norm["timestamp"] = parsed
        elif isinstance(ts, (int, float)):
            try:
                norm["timestamp"] = dt.datetime.fromtimestamp(float(ts))
            except Exception:
                norm["timestamp"] = None

        # gpu index
        if "gpu_index" in norm and isinstance(norm["gpu_index"], str):
            norm["gpu_index"] = _extract_int(str(norm["gpu_index"]))

        # utilization (keep for generic sources)
        if "utilization" in norm:
            val = norm["utilization"]
            if isinstance(val, str):
                val = val.strip().rstrip("%")
                v = _extract_float(val)
                if v is not None and 0.0 <= v <= 1.0:
                    v *= 100.0
                norm["utilization"] = v
            elif isinstance(val, (int, float)):
                v = float(val)
                if 0.0 <= v <= 1.0:
                    v *= 100.0
                norm["utilization"] = v

        # memory_percent
        if "memory_percent" in norm:
            val = norm["memory_percent"]
            if isinstance(val, str):
                s = val.strip().rstrip("%")
                v = _extract_float(s)
                if v is not None and 0.0 <= v <= 1.0:
                    v *= 100.0
                norm["memory_percent"] = v
            elif isinstance(val, (int, float)):
                v = float(val)
                if 0.0 <= v <= 1.0:
                    v *= 100.0
                norm["memory_percent"] = v

        # memory
        if "memory_used_mb" in norm:
            val = norm["memory_used_mb"]
            if isinstance(val, str):
                s = val.strip().lower().rstrip("b")
                s = s.replace("mib", "").replace("mb", "")
                s = s.replace("gib", "").replace("gb", "")
                f = _extract_float(s)
                if f is not None and ("gi" in val.lower() or "gb" in val.lower()):
                    f *= 1024.0
                norm["memory_used_mb"] = f

        # processes to task_count
        if "processes" in norm and isinstance(norm["processes"], list):
            if "task_count" not in norm:
                norm["task_count"] = len(norm["processes"])  # type: ignore[assignment]

        # coalesce: if only utilization present from generic sources, mirror to memory_percent
        if "memory_percent" not in norm and "utilization" in norm:
            norm["memory_percent"] = norm.get("utilization")

        # minimal fields required
        if not norm.get("timestamp") or norm.get("gpu_index") is None:
            return None

        return {
            "timestamp": norm.get("timestamp"),
            "gpu_index": norm.get("gpu_index"),
            "memory_percent": norm.get("memory_percent"),
            "task_count": norm.get("task_count"),
            "utilization": norm.get("utilization"),  # kept for compatibility in CSV
            "memory_used_mb": norm.get("memory_used_mb"),
        }

    def _parse_json(self, line: str) -> List[Dict[str, object]]:
        if not (line.startswith("{") and line.endswith("}")):
            # not strictly required, but helps avoid false positives
            pass
        obj = json.loads(line)
        if isinstance(obj, dict):
            # If dict contains a gpus array, expand
            if "gpus" in obj and isinstance(obj["gpus"], list):
                out: List[Dict[str, object]] = []
                for g in obj["gpus"]:
                    if not isinstance(g, dict):
                        continue
                    # Merge GPU dict directly; timestamp must be attached by caller/other parser
                    rec = self._normalize_record(g)
                    if rec:
                        out.append(rec)
                return out
            rec = self._normalize_record(obj)
            return [rec] if rec else []
        if isinstance(obj, list):
            out: List[Dict[str, object]] = []
            for item in obj:
                if isinstance(item, dict):
                    rec = self._normalize_record(item)
                    if rec:
                        out.append(rec)
            return out
        return []

    def _parse_ts_plus_json(self, line: str) -> List[Dict[str, object]]:
        # Pattern: "<timestamp> {json}"
        m = re.match(r"^(?P<ts>[^\{]+?)\s*(?P<js>\{.*\})\s*$", line)
        if not m:
            return []
        ts_raw = m.group("ts").strip()
        ts = _parse_timestamp(ts_raw)
        if not ts:
            return []
        js = m.group("js")
        try:
            obj = json.loads(js)
        except Exception:
            return []
        out: List[Dict[str, object]] = []
        if isinstance(obj, dict) and isinstance(obj.get("gpus"), list):
            for g in obj["gpus"]:
                if not isinstance(g, dict):
                    continue
                # Build record combining timestamp + gpu fields
                rec: Dict[str, object] = {"timestamp": ts}
                rec.update(g)
                norm = self._normalize_record(rec)
                if norm:
                    out.append(norm)
            return out
        # Fallback: treat as a single record with timestamp + dict
        if isinstance(obj, dict):
            rec = {"timestamp": ts}
            rec.update(obj)
            norm = self._normalize_record(rec)
            return [norm] if norm else []
        return []

    def _parse_kv(self, line: str) -> List[Dict[str, object]]:
        # Example: ts=2025-10-21 12:34:56.123 gpu=0 util=34% mem=1234MiB pids=3
        pairs = self.KV_PATTERN.findall(line)
        if not pairs:
            return []
        rec: Dict[str, object] = {}
        for k, v in pairs:
            k_norm = self.KEY_ALIASES.get(k, k)
            if k_norm == "processes":
                # processes are not representable via single token; skip
                continue
            rec[k_norm] = v
        # pid count heuristic: count pid= occurrences in line
        pid_inline = len(re.findall(r"pid\s*=", line))
        if pid_inline > 0 and "pid_count" not in rec:
            rec["pid_count"] = pid_inline
        out = self._normalize_record(rec)
        return [out] if out else []

    def _parse_csv(self, line: str) -> List[Dict[str, object]]:
        # Try to detect header first
        if self.header_keys is None:
            # header if contains non-numeric keys like timestamp,index,utilization.gpu
            parts = [p.strip() for p in line.split(",")]
            if any(re.search(r"[A-Za-z]", p) for p in parts):
                self.header_keys = [self.KEY_ALIASES.get(p, p) for p in parts]
                return []
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            return []
        rec: Dict[str, object] = {}
        if self.header_keys:
            for k, v in zip(self.header_keys, parts):
                rec[k] = v
        else:
            # attempt positional mapping like: ts,index,util,...
            candidates = [
                ("timestamp", parts[0] if len(parts) > 0 else None),
                ("gpu_index", parts[1] if len(parts) > 1 else None),
                ("utilization", parts[2] if len(parts) > 2 else None),
            ]
            for k, v in candidates:
                if v is not None:
                    rec[k] = v
        out = self._normalize_record(rec)
        return [out] if out else []

    def _parse_fallback_gpu_block(self, line: str) -> List[Dict[str, object]]:
        # Fallback for lines like: "2025-10-21 12:34:56 GPU 0: 35% | GPU 1: 22% | ..."
        # Extract timestamp at start
        m = re.match(r"^([^|]+?)\s+GPU\s*\d+", line)
        if not m:
            return []
        ts_str = m.group(1).strip()
        ts = _parse_timestamp(ts_str)
        if not ts:
            return []
        out: List[Dict[str, object]] = []
        for gm in self.GPU_UTIL_PATTERN.finditer(line):
            gpu_idx = _extract_int(gm.group(1))
            util = _extract_float(gm.group(2))
            if gpu_idx is None or util is None:
                continue
            out.append({
                "timestamp": ts,
                "gpu_index": gpu_idx,
                "utilization": util,
            })
        return out


def read_log(path: str, forced_format: Optional[str]) -> List[Dict[str, object]]:
    parser = LineParser(forced_format)
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            recs = parser.parse_line(line)
            if not recs:
                continue
            for r in recs:
                if r and r.get("timestamp") and r.get("gpu_index") is not None:
                    rows.append(r)
    return rows


def write_csv(rows: List[Dict[str, object]], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fields = [
        "timestamp",
        "gpu_index",
        "memory_percent",
        "task_count",
        "utilization",
        "memory_used_mb",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "timestamp": (r.get("timestamp").isoformat(sep=" ") if isinstance(r.get("timestamp"), dt.datetime) else r.get("timestamp")),
                "gpu_index": r.get("gpu_index"),
                "memory_percent": r.get("memory_percent"),
                "task_count": r.get("task_count"),
                "utilization": r.get("utilization"),
                "memory_used_mb": r.get("memory_used_mb"),
            })


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception as e:
        print("[warn] matplotlib not available, skip PNG charts:", e, file=sys.stderr)
        return None


def _import_plotly():
    try:
        import plotly.graph_objs as go  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore
        return go, make_subplots
    except Exception:
        return None, None


def build_series(rows: List[Dict[str, object]]) -> Dict[int, List[Tuple[dt.datetime, Optional[float], Optional[int]]]]:
    by_gpu: Dict[int, List[Tuple[dt.datetime, Optional[float], Optional[int]]]] = defaultdict(list)
    for r in rows:
        gpu = int(r["gpu_index"])  # type: ignore[arg-type]
        ts = r["timestamp"]
        if not isinstance(ts, dt.datetime):
            continue
        util = r.get("memory_percent")
        util_f = float(util) if util is not None else None
        pidc = r.get("task_count")
        pid_i = int(pidc) if pidc is not None else None
        by_gpu[gpu].append((ts, util_f, pid_i))
    # sort by time
    for g in by_gpu:
        by_gpu[g].sort(key=lambda t: t[0])
    return by_gpu


def plot_png(by_gpu: Dict[int, List[Tuple[dt.datetime, Optional[float], Optional[int]]]], out_dir: str) -> bool:
    plt = _import_matplotlib()
    if plt is None:
        return False
    os.makedirs(out_dir, exist_ok=True)
    gpus = sorted(by_gpu.keys())
    if not gpus:
        print("[warn] No GPU data parsed; skip PNG charts")
        return False
    n = len(gpus)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(3.0, 2.2 * n)), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, gpu in zip(axes, gpus):
        series = by_gpu[gpu]
        times = [t for t, _, _ in series]
        utils = [u if u is not None else float('nan') for _, u, _ in series]
        pidcs = [p if p is not None else float('nan') for _, _, p in series]
        ax.plot(times, utils, label=f"GPU {gpu} memory %", color="#1f77b4", linewidth=1.2)
        ax.set_ylabel(f"GPU {gpu} %")
        ax.grid(True, alpha=0.3)
        # Add pid count on twin axis if present
        if any(p is not None for p in pidcs):
            ax2 = ax.twinx()
            ax2.step(times, pidcs, where="post", label="tasks", color="#ff7f0e", alpha=0.7)
            ax2.set_ylabel("tasks")
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("time")
    fig.suptitle("GPU memory% and task counts")
    fig.autofmt_xdate()
    out_png = os.path.join(out_dir, "gpu_utilization.png")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # total task concurrency chart if pid_count available
    has_pid = any(any(p is not None for _, _, p in by_gpu[g]) for g in gpus)
    if has_pid:
        import math
        # build total per timestamp by merging unique timestamps
        all_times = sorted(set(t for g in gpus for t, _, _ in by_gpu[g]))
        total: List[Tuple[dt.datetime, float]] = []
        for ts in all_times:
            s = 0
            for g in gpus:
                # find pid count at ts (exact match); if none, carry last known
                series = by_gpu[g]
                last = None
                for t, _, p in series:
                    if t <= ts and p is not None:
                        last = p
                    elif t > ts:
                        break
                s += last or 0
            total.append((ts, float(s)))
        fig2, ax = plt.subplots(1, 1, figsize=(14, 3.5))
        ax.step([t for t, _ in total], [v for _, v in total], where="post", color="#2ca02c")
        ax.set_title("Total concurrent tasks")
        ax.set_ylabel("tasks")
        ax.set_xlabel("time")
        ax.grid(True, alpha=0.3)
        fig2.autofmt_xdate()
        out_png2 = os.path.join(out_dir, "total_tasks.png")
        fig2.tight_layout()
        fig2.savefig(out_png2, dpi=150)
        plt.close(fig2)

    return True


def plot_png_plotly(by_gpu: Dict[int, List[Tuple[dt.datetime, Optional[float], Optional[int]]]], out_dir: str) -> bool:
    # Fallback: use plotly + kaleido to write PNGs if matplotlib is unavailable
    go, make_subplots = _import_plotly()
    if go is None or make_subplots is None:
        return False
    try:
        import plotly.io as pio  # type: ignore
    except Exception:
        return False
    os.makedirs(out_dir, exist_ok=True)
    gpus = sorted(by_gpu.keys())
    if not gpus:
        return False
    fig = make_subplots(rows=len(gpus), cols=1, shared_xaxes=True, subplot_titles=[f"GPU {g}" for g in gpus])
    for i, g in enumerate(gpus, start=1):
        series = by_gpu[g]
        times = [t for t, _, _ in series]
        utils = [u if u is not None else None for _, u, _ in series]
        pidcs = [p if p is not None else None for _, _, p in series]
        fig.add_trace(go.Scatter(x=times, y=utils, name=f"GPU {g} memory%", mode="lines"), row=i, col=1)
        if any(p is not None for p in pidcs):
            fig.add_trace(go.Scatter(x=times, y=pidcs, name=f"GPU {g} tasks", mode="lines"), row=i, col=1)
    fig.update_layout(height=max(300, 220 * len(gpus)), title_text="GPU memory% and tasks")
    out_png = os.path.join(out_dir, "gpu_utilization.png")
    try:
        # requires kaleido
        pio.write_image(fig, out_png, format="png", scale=2)
        # Also write total_tasks if available
        has_pid = any(any(p is not None for _, _, p in by_gpu[g]) for g in gpus)
        if has_pid:
            import numpy as np  # type: ignore
            all_times = sorted(set(t for g in gpus for t, _, _ in by_gpu[g]))
            totals = []
            for ts in all_times:
                s = 0
                for g in gpus:
                    series = by_gpu[g]
                    last = None
                    for t, _, p in series:
                        if t <= ts and p is not None:
                            last = p
                        elif t > ts:
                            break
                    s += last or 0
                totals.append((ts, float(s)))
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=[t for t, _ in totals], y=[v for _, v in totals], mode="lines", name="total tasks"))
            fig2.update_layout(title="Total concurrent tasks", xaxis_title="time", yaxis_title="tasks")
            pio.write_image(fig2, os.path.join(out_dir, "total_tasks.png"), format="png", scale=2)
        return True
    except Exception as e:
        print("[warn] plotly+kaleido PNG export failed:", e, file=sys.stderr)
        return False


def plot_html(by_gpu: Dict[int, List[Tuple[dt.datetime, Optional[float], Optional[int]]]], out_dir: str) -> bool:
    go, make_subplots = _import_plotly()
    if go is None or make_subplots is None:
        print("[warn] plotly not available; skip HTML charts", file=sys.stderr)
        return False
    os.makedirs(out_dir, exist_ok=True)
    gpus = sorted(by_gpu.keys())
    if not gpus:
        return False
    fig = make_subplots(rows=len(gpus), cols=1, shared_xaxes=True, subplot_titles=[f"GPU {g}" for g in gpus])
    for i, g in enumerate(gpus, start=1):
        series = by_gpu[g]
        times = [t for t, _, _ in series]
        utils = [u if u is not None else None for _, u, _ in series]
        pidcs = [p if p is not None else None for _, _, p in series]
        fig.add_trace(go.Scatter(x=times, y=utils, name=f"GPU {g} memory%", mode="lines"), row=i, col=1)
        if any(p is not None for p in pidcs):
            fig.add_trace(go.Scatter(x=times, y=pidcs, name=f"GPU {g} tasks", mode="lines"), row=i, col=1)
    fig.update_layout(height=max(300, 220 * len(gpus)), title_text="GPU memory% and tasks")
    out_html = os.path.join(out_dir, "gpu_utilization.html")
    try:
        # Use inline plotly.js so the file works offline via file:// protocol
        fig.write_html(out_html, include_plotlyjs="inline")
        return True
    except Exception as e:
        print("[warn] failed to write HTML:", e, file=sys.stderr)
        return False


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Analyze GPU monitor logs and generate charts")
    p.add_argument("--input", required=True, help="Path to monitor.log or similar")
    p.add_argument("--output-dir", required=False, default=None, help="Directory to write CSV and charts")
    p.add_argument("--format", choices=["auto", "json", "kv", "csv"], default="auto", help="Hint the parser format")
    p.add_argument("--no-png", action="store_true", help="Disable PNG chart generation (matplotlib)")
    p.add_argument("--html", action="store_true", help="Generate interactive HTML charts (plotly if available)")
    args = p.parse_args(argv)

    in_path = os.path.abspath(args.input)
    if not os.path.exists(in_path):
        print(f"[error] input not found: {in_path}", file=sys.stderr)
        return 2
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(os.path.dirname(in_path), "monitor_plots")
    os.makedirs(out_dir, exist_ok=True)

    forced = None if args.format == "auto" else args.format
    print(f"[info] reading: {in_path}")
    rows = read_log(in_path, forced)
    if not rows:
        print("[error] No rows parsed. Try --format hint or share a sample line.", file=sys.stderr)
        return 3
    out_csv = os.path.join(out_dir, "parsed_gpu_metrics.csv")
    write_csv(rows, out_csv)
    print(f"[info] wrote CSV: {out_csv} ({len(rows)} rows)")

    by_gpu = build_series(rows)
    if not args.no_png:
        if plot_png(by_gpu, out_dir) or plot_png_plotly(by_gpu, out_dir):
            print(f"[info] wrote PNG charts in: {out_dir}")
    if args.html:
        if plot_html(by_gpu, out_dir):
            print(f"[info] wrote HTML charts in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# PLBACKEND=Agg python /data/apps/project/cu2tri/tools/gpu_log_analyzer.py --input /data/apps/project/cu2tri/logs/monitor.log --output-dir /data/apps/project/cu2tri/logs/monitor_plots