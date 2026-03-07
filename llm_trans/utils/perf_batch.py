# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Lazy import NVGPU client via llm_trans wrapper to respect PROJECT_ROOT
def _load_nvgpu_client():
    try:
        from ..clients.nvgpu import NVGPU_AVAILABLE, NVGPUClient  # type: ignore
        return NVGPU_AVAILABLE, NVGPUClient
    except Exception:
        return False, None


@dataclass
class Target:
    script_path: Path
    attempt_dir: Path
    case_tag: str
    shape_tag: Optional[str] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch run perf-only tests for existing attempts")
    p.add_argument("--runs", required=True, help="Path to a specific runs/<timestamp> directory")
    p.add_argument("--mode", choices=["nvgpu", "local"], default="nvgpu", help="Execution mode")
    p.add_argument("--gpu", type=int, nargs="*", default=[], help="Preferred GPU IDs (NVGPU mode)")
    p.add_argument("--server", default="http://localhost:8080", help="NVGPU server URL")
    p.add_argument("--concurrency", type=int, default=8, help="Max concurrent submissions/runs")
    p.add_argument("--pattern", default=None, help="Optional glob pattern under runs to filter attempts")
    p.add_argument(
        "--case-type",
        nargs="*",
        default=None,
        help="Filter case directory names (glob). Examples: avgpool_* add_*",
    )
    p.add_argument(
        "--attempt-index",
        nargs="*",
        default=None,
        help="Filter attempt indices. Examples: 1 3 10 (-> attempt_01/03/10)",
    )
    p.add_argument("--force", action="store_true", help="Re-run even if perf.json exists")
    p.add_argument("--out-jsonl", default=None, help="Output JSONL path (default: <runs>/perf.jsonl)")
    p.add_argument("--out-csv", default=None, help="Optional CSV path for summary table")
    p.add_argument("--fallback-local", action="store_true", help="When NVGPU is unhealthy, fallback to local mode")
    p.add_argument(
        "--include-failed",
        action="store_true",
        help="Include attempts without a passing triton_test_round log",
    )
    return p.parse_args()


_TEST_LOG_PATTERN = re.compile(r"triton_test_round_(\d+)\.log$")


def _latest_triton_test_log(logs_dir: Path) -> Optional[Path]:
    latest_round = -1
    latest_path: Optional[Path] = None
    for log_path in logs_dir.glob("triton_test_round_*.log"):
        match = _TEST_LOG_PATTERN.match(log_path.name)
        if not match:
            continue
        try:
            round_num = int(match.group(1))
        except ValueError:
            continue
        if round_num > latest_round:
            latest_round = round_num
            latest_path = log_path
    return latest_path


def _attempt_has_passing_log(attempt_dir: Path) -> Tuple[bool, Optional[str]]:
    logs_dir = attempt_dir / "logs"
    if not logs_dir.exists():
        return False, f"[skip] {attempt_dir}: logs directory not found"

    latest_log = _latest_triton_test_log(logs_dir)
    if latest_log is None:
        return False, f"[skip] {attempt_dir}: no triton_test_round_*.log found"

    try:
        content = latest_log.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        return False, f"[skip] {attempt_dir}: failed to read {latest_log.name} ({exc})"

    if "STATUS: PASSED" in content:
        return True, None

    return False, f"[skip] {attempt_dir}: latest log {latest_log.name} not marked PASSED"


def _normalize_attempt_name(s: str) -> Optional[str]:
    s = s.strip()
    if not s:
        return None
    if s.startswith("attempt_"):
        return s
    try:
        idx = int(s)
        return f"attempt_{idx:02d}"
    except Exception:
        return None


def _find_targets(
    runs_dir: Path,
    pattern: Optional[str],
    *,
    include_failed: bool,
    case_type_filters: Optional[List[str]] = None,
    attempt_index_filters: Optional[List[str]] = None,
) -> Tuple[List[Target], List[str]]:
    if pattern:
        candidates = runs_dir.glob(pattern)
        script_paths = [
            Path(p)
            for p in candidates
            if Path(p).is_file() and Path(p).name.startswith("check_triton") and Path(p).suffix == ".py"
        ]
    else:
        script_paths = list(runs_dir.glob("**/attempt_*/check_triton*.py"))

    targets: List[Target] = []
    skipped: List[str] = []

    attempt_dir_names: Optional[set[str]] = None
    if attempt_index_filters:
        attempt_dir_names = set()
        for raw in attempt_index_filters:
            norm = _normalize_attempt_name(str(raw))
            if norm:
                attempt_dir_names.add(norm)

    for script in script_paths:
        attempt_dir = script.parent
        case_dir = attempt_dir.parent
        case_name = case_dir.name

        # case-type filter (glob)
        if case_type_filters:
            if not any(fnmatch.fnmatch(case_name, pat) for pat in case_type_filters):
                skipped.append(f"[skip] {attempt_dir}: case '{case_name}' not matched by --case-type")
                continue

        # attempt-index filter
        if attempt_dir_names is not None and attempt_dir.name not in attempt_dir_names:
            skipped.append(f"[skip] {attempt_dir}: attempt '{attempt_dir.name}' not in --attempt-index")
            continue
        if not include_failed:
            passed, reason = _attempt_has_passing_log(attempt_dir)
            if not passed:
                if reason:
                    skipped.append(reason)
                continue

        # case tag: parent dir name
        case_tag = case_name
        targets.append(Target(script_path=script, attempt_dir=attempt_dir, case_tag=case_tag, shape_tag=None))
    return targets, skipped


def _perf_json_path(attempt_dir: Path) -> Path:
    return attempt_dir / "perf.json"


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    import csv
    headers = [
        "case",
        "shape",
        "gpu",
        "triton_ms",
        "cuda_ms",
        "torch_ms",
        "speedup_cuda_vs_torch",
        "speedup_triton_vs_cuda",
        "speedup_triton_vs_torch",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for rec in records:
            results_ms = rec.get("results_ms", {})
            speedup = rec.get("speedup", {})
            writer.writerow([
                rec.get("case"),
                rec.get("shape"),
                rec.get("gpu"),
                results_ms.get("triton"),
                results_ms.get("cuda"),
                results_ms.get("torch"),
                speedup.get("cuda_vs_torch"),
                speedup.get("triton_vs_cuda"),
                speedup.get("triton_vs_torch"),
            ])


def _save_perf_artifacts(
    target: Target,
    perf_record: Dict[str, Any],
    server_times: Optional[Dict[str, Any]],
    logs: Dict[str, str],
) -> None:
    logs_dir = target.attempt_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    enriched_record: Dict[str, Any] = {**perf_record}
    if server_times:
        enriched_record["server_times_ms"] = server_times

    latest_path = logs_dir / "perf_latest.json"
    latest_path.write_text(json.dumps(enriched_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    history_path = logs_dir / "perf_history.jsonl"
    with history_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(enriched_record, ensure_ascii=False) + "\n")

    stdout_path = logs_dir / "perf_task_stdout.log"
    stderr_path = logs_dir / "perf_task_stderr.log"
    if logs.get("stdout"):
        stdout_path.write_text(logs["stdout"], encoding="utf-8")
    if logs.get("stderr"):
        stderr_path.write_text(logs["stderr"], encoding="utf-8")


async def _run_local(target: Target) -> Tuple[Optional[Dict[str, Any]], Dict[str, str], Optional[str]]:
    perf_out = _perf_json_path(target.attempt_dir)
    cmd = [
        sys.executable,
        target.script_path.name,
        "--perf-only",
        "--perf-json-out",
        str(perf_out),
        "--case-tag",
        target.case_tag,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(target.attempt_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_data, stderr_data = await proc.communicate()
    stdout_text = stdout_data.decode("utf-8", errors="ignore")
    stderr_text = stderr_data.decode("utf-8", errors="ignore")
    if proc.returncode != 0:
        return None, {"stdout": stdout_text, "stderr": stderr_text}, stderr_text or f"exit code {proc.returncode}"
    try:
        if perf_out.exists():
            return json.loads(perf_out.read_text(encoding="utf-8")), {"stdout": stdout_text, "stderr": stderr_text}, None
        # fallback stdout parse
        lines = [ln for ln in stdout_text.splitlines() if ln.strip()]
        data = json.loads(lines[-1]) if lines else None
        return data, {"stdout": stdout_text, "stderr": stderr_text}, None
    except Exception as e:
        return None, {"stdout": stdout_text, "stderr": stderr_text}, f"parse error: {e}"


async def _run_nvgpu(
    target: Target,
    server: str,
    gpu_id: Optional[int],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str], Dict[str, str]]:
    NVGPU_AVAILABLE, NVGPUClient = _load_nvgpu_client()
    if not NVGPU_AVAILABLE or NVGPUClient is None:
        return None, None, "NVGPU client unavailable", {"stdout": "", "stderr": ""}
    client = NVGPUClient(server)
    if not client.health_check():
        return None, None, f"NVGPU server unhealthy: {server}", {"stdout": "", "stderr": ""}

    perf_out = _perf_json_path(target.attempt_dir)
    args = [
        "--perf-only",
        "--perf-json-out",
        str(perf_out),
        "--case-tag",
        target.case_tag,
    ]
    task_id = client.submit_task_in_script_dir(
        script_path=str(target.script_path.absolute()),
        task_mode="exclusive",
        task_type="performance",
        task_label=target.case_tag,
        args=args,
        gpu_id=gpu_id,
    )

    # poll
    while True:
        result = client.get_task(task_id)
        if result.status in ["completed", "failed", "cancelled"]:
            break
        await asyncio.sleep(2)

    if result.status != "completed" or (result.exit_code is not None and result.exit_code != 0):
        stderr = client.get_full_task_log(task_id, log_type="stderr")
        stdout = client.get_full_task_log(task_id, log_type="stdout")
        return None, None, f"task {task_id} failed: status={result.status} exit={result.exit_code} err_tail={stderr[-800:] if stderr else ''}", {"stdout": stdout or "", "stderr": stderr or ""}

    try:
        if perf_out.exists():
            data = json.loads(perf_out.read_text(encoding="utf-8"))
        else:
            stdout = client.get_full_task_log(task_id, log_type="stdout")
            lines = [ln for ln in stdout.splitlines() if ln.strip()]
            data = json.loads(lines[-1]) if lines else None
    except Exception as e:
        stdout = client.get_full_task_log(task_id, log_type="stdout")
        stderr = client.get_full_task_log(task_id, log_type="stderr")
        return None, None, f"parse error: {e}", {"stdout": stdout or "", "stderr": stderr or ""}

    stdout_content = client.get_full_task_log(task_id, log_type="stdout")
    stderr_content = client.get_full_task_log(task_id, log_type="stderr")

    server_times: Dict[str, Any] = {
        "pending_ms": result.pending_duration_ms,
        "queue_ms": result.queue_duration_ms,
        "waiting_ms": result.waiting_duration_ms,
        "running_ms": result.running_duration_ms,
        "total_ms": result.total_duration_ms,
        "execution_ms": result.execution_duration_ms,
        "assigned_gpu": result.gpu_id,
        "status": result.status,
        "exit_code": result.exit_code,
        "task_id": result.task_id,
    }
    return data, server_times, None, {"stdout": stdout_content or "", "stderr": stderr_content or ""}


async def _bounded_worker(sem: asyncio.Semaphore, coro):
    async with sem:
        return await coro


async def run_batch(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    runs_dir = Path(args.runs).expanduser().resolve()
    if not runs_dir.exists():
        raise RuntimeError(f"Runs directory not found: {runs_dir}")

    targets, skipped = _find_targets(
        runs_dir,
        args.pattern,
        include_failed=args.include_failed,
        case_type_filters=args.case_type,
        attempt_index_filters=args.attempt_index,
    )

    if skipped:
        print(f"Skipped {len(skipped)} attempt(s) without a passing log:")
        for msg in skipped:
            print(msg)

    if not targets:
        print("No attempts found to run.")
        return [], []

    print(f"Scheduling {len(targets)} attempt(s) for performance measurement...")

    # filter by perf.json existence unless force
    if not args.force:
        targets = [t for t in targets if not _perf_json_path(t.attempt_dir).exists()]

    if not targets:
        print("Nothing to do (all attempts already have perf.json). Use --force to re-run.")
        return [], []

    # GPU rotation
    gpu_list: List[int] = list(args.gpu) if args.gpu else []
    gpu_idx: int = 0

    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    sem = asyncio.Semaphore(max(1, int(args.concurrency)))
    tasks = []

    for tgt in targets:
        if args.mode == "nvgpu":
            selected_gpu = None
            if gpu_list:
                selected_gpu = gpu_list[gpu_idx % len(gpu_list)]
                gpu_idx += 1
            tasks.append(_bounded_worker(sem, _run_nvgpu(tgt, args.server, selected_gpu)))
        else:
            tasks.append(_bounded_worker(sem, _run_local(tgt)))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for tgt, res in zip(targets, results):
        if isinstance(res, Exception):
            errors.append({
                "_type": "perf_error",
                "case": tgt.case_tag,
                "attempt_dir": str(tgt.attempt_dir),
                "error": str(res),
            })
            continue

        if args.mode == "nvgpu":
            data, server_times, err, task_logs = res  # type: ignore
            if err or not data:
                errors.append({
                    "_type": "perf_error",
                    "case": tgt.case_tag,
                    "attempt_dir": str(tgt.attempt_dir),
                    "error": err or "no data",
                })
                continue
            rec = {"_type": "perf", **data, "server_times_ms": server_times or {}}
            records.append(rec)
            _save_perf_artifacts(tgt, rec, server_times, task_logs)
            print(f"[perf] Completed {tgt.case_tag} ({tgt.attempt_dir.name}) via NVGPU")
        else:
            data, task_logs, err = res  # type: ignore
            if err or not data:
                errors.append({
                    "_type": "perf_error",
                    "case": tgt.case_tag,
                    "attempt_dir": str(tgt.attempt_dir),
                    "error": err or "no data",
                })
                continue
            rec = {"_type": "perf", **data}
            records.append(rec)
            _save_perf_artifacts(tgt, rec, None, task_logs)
            print(f"[perf] Completed {tgt.case_tag} ({tgt.attempt_dir.name}) locally")

    return records, errors


def main() -> None:
    args = parse_args()

    if args.mode == "nvgpu":
        NVGPU_AVAILABLE, NVGPUClient = _load_nvgpu_client()
        if not NVGPU_AVAILABLE or NVGPUClient is None:
            if args.fallback_local:
                print("[warn] NVGPU unavailable; falling back to local mode")
                args.mode = "local"
            else:
                raise SystemExit("NVGPU client unavailable. Use --mode local or --fallback-local.")

    records, errors = asyncio.run(run_batch(args))

    out_jsonl = Path(args.out_jsonl).resolve() if args.out_jsonl else Path(args.runs).resolve() / "perf.jsonl"
    if records or errors:
        _write_jsonl(out_jsonl, records + errors)
        print(f"Wrote {len(records)} perf records (+{len(errors)} errors) to {out_jsonl}")
    else:
        print("No records written.")

    if args.out_csv and records:
        out_csv = Path(args.out_csv).resolve()
        _write_csv(out_csv, records)
        print(f"Wrote CSV summary to {out_csv}")


if __name__ == "__main__":
    main()


