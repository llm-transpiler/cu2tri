# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect perf results under runs/<timestamp> directory")
    p.add_argument("--runs", required=True, help="Path to a specific runs/<timestamp> directory")
    p.add_argument("--out-jsonl", default=None, help="Output JSONL path (default: <runs>/perf.jsonl)")
    p.add_argument("--out-csv", default=None, help="Optional CSV path for summary table")
    return p.parse_args()


def find_perf_json_files(root: Path) -> List[Path]:
    return list(root.glob("**/attempt_*/perf.json"))


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    import csv

    # Build simple flat table
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


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs).expanduser().resolve()
    if not runs_dir.exists():
        raise SystemExit(f"Runs directory not found: {runs_dir}")

    perf_files = find_perf_json_files(runs_dir)
    records: List[Dict[str, Any]] = []
    for p in perf_files:
        data = load_json(p)
        if data:
            records.append(data)

    if not records:
        print("No perf.json found under", runs_dir)
        return

    out_jsonl = Path(args.out_jsonl).resolve() if args.out_jsonl else runs_dir / "perf.jsonl"
    write_jsonl(out_jsonl, records)
    print(f"Wrote {len(records)} records to {out_jsonl}")

    if args.out_csv:
        out_csv = Path(args.out_csv).resolve()
        write_csv(out_csv, records)
        print(f"Wrote CSV summary to {out_csv}")


if __name__ == "__main__":
    main()
