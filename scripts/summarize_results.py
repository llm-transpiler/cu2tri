#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class MatmulRecord:
    k: int
    latency_s: float
    tflops: float
    gpu: str


@dataclass
class MambaRecord:
    seq_len: int
    latency_ms: float
    tflops: float
    gpu: str


MATMUL_LINE_RE = re.compile(r"^K=\s*(?P<k>\d+)\s+latency=(?P<lat>[0-9.]+)s\s+TFlops=(?P<tflops>[0-9.]+)")
MAMBA_LINE_RE = re.compile(r"^seq_len=\s*(?P<seq>\d+)\s+latency=(?P<lat>[0-9.]+)ms\s+TFlops=(?P<tflops>[0-9.]+)")


def find_results_roots(base: Path) -> List[Path]:
    results_root = base / "results"
    roots: List[Path] = []
    if results_root.exists():
        roots.append(results_root)
    docker_root = results_root / "docker"
    if docker_root.exists():
        for sub in docker_root.iterdir():
            if sub.is_dir():
                roots.append(sub)
    unique: List[Path] = []
    seen = set()
    for r in roots:
        rp = r.resolve()
        if rp not in seen:
            unique.append(rp)
            seen.add(rp)
    return unique


def parse_gpu_id_from_path(p: Path) -> str:
    try:
        return p.parent.parent.name
    except Exception:
        return "gpu?"


def parse_matmul_log(log_path: Path) -> Dict[int, MatmulRecord]:
    gpu = parse_gpu_id_from_path(log_path)
    per_k: Dict[int, MatmulRecord] = {}
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = MATMUL_LINE_RE.search(line)
                if not m:
                    continue
                k = int(m.group("k"))
                lat_s = float(m.group("lat"))
                tflops = float(m.group("tflops"))
                per_k[k] = MatmulRecord(k=k, latency_s=lat_s, tflops=tflops, gpu=gpu)
    except FileNotFoundError:
        pass
    return per_k


def parse_mamba_log(log_path: Path) -> Dict[int, MambaRecord]:
    gpu = parse_gpu_id_from_path(log_path)
    per_seq: Dict[int, MambaRecord] = {}
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = MAMBA_LINE_RE.search(line)
                if not m:
                    continue
                seq = int(m.group("seq"))
                lat_ms = float(m.group("lat"))
                tflops = float(m.group("tflops"))
                per_seq[seq] = MambaRecord(seq_len=seq, latency_ms=lat_ms, tflops=tflops, gpu=gpu)
    except FileNotFoundError:
        pass
    return per_seq


def aggregate_best_matmul(records_per_gpu: List[Dict[int, MatmulRecord]]) -> Dict[int, MatmulRecord]:
    best: Dict[int, MatmulRecord] = {}
    for per_gpu in records_per_gpu:
        for k, rec in per_gpu.items():
            if k not in best or rec.tflops > best[k].tflops:
                best[k] = rec
    return best


def aggregate_best_mamba(records_per_gpu: List[Dict[int, MambaRecord]]) -> Dict[int, MambaRecord]:
    best: Dict[int, MambaRecord] = {}
    for per_gpu in records_per_gpu:
        for seq, rec in per_gpu.items():
            if seq not in best or rec.tflops > best[seq].tflops:
                best[seq] = rec
    return best


def matmul_table_lines(best: Dict[int, MatmulRecord]) -> List[str]:
    lines: List[str] = []
    lines.append("| K     | Latency (s) | Throughput (TFLOPs) |")
    lines.append("|-------|-------------|---------------------|")
    for k in sorted(best.keys()):
        rec = best[k]
        throughput_int = int(round(rec.tflops * 1000))
        lines.append(f"| {k:5d} | {rec.latency_s:0.6f}    | {throughput_int:<19d} |")
    return lines


def mamba_table_lines(best: Dict[int, MambaRecord]) -> List[str]:
    lines: List[str] = []
    lines.append("| Seq_len| Latency (ms) | Throughput (TFLOPs) |")
    lines.append("|-------|-------------|---------------------|")
    for seq in sorted(best.keys()):
        rec = best[seq]
        lines.append(f"| {seq:5d} | {rec.latency_ms:0.3f}    | {rec.tflops:0.3f}                 |")
    return lines


def replace_results_table(readme_path: Path, new_table_lines: List[str]) -> None:
    if not readme_path.exists():
        return
    text = readme_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    hdr_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("## results"):
            hdr_idx = i
            break
    if hdr_idx is None:
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.append("## Results")
        lines.append("")
        lines.extend(new_table_lines)
        lines.append("")
        readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    start = None
    j = hdr_idx + 1
    while j < len(lines) and lines[j].strip() == "":
        j += 1
    if j < len(lines) and lines[j].lstrip().startswith("|"):
        start = j
    else:
        insert_at = j
        new_lines = lines[:insert_at] + new_table_lines + [""] + lines[insert_at:]
        readme_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return

    end = start
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    new_lines = lines[:start] + new_table_lines + lines[end:]
    readme_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def print_matmul_table(title: str, best: Dict[int, MatmulRecord]) -> None:
    if not best:
        return
    print(f"## {title}")
    print()
    for l in matmul_table_lines(best):
        print(l)
    print()


def print_mamba_table(title: str, best: Dict[int, MambaRecord]) -> None:
    if not best:
        return
    print(f"## {title}")
    print()
    for l in mamba_table_lines(best):
        print(l)
    print()

# --------- Stats (mean/median) across GPUs ---------

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def median(values: List[float]) -> float:
    n = len(values)
    if n == 0:
        return float("nan")
    s = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def compute_matmul_stats(per_gpu: Dict[str, Dict[int, MatmulRecord]]) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Tuple[float, float]]]:
    by_k: Dict[int, List[MatmulRecord]] = {}
    for gpu_map in per_gpu.values():
        for k, rec in gpu_map.items():
            by_k.setdefault(k, []).append(rec)
    mean_map: Dict[int, Tuple[float, float]] = {}
    median_map: Dict[int, Tuple[float, float]] = {}
    for k, recs in by_k.items():
        lats = [r.latency_s for r in recs]
        tfs = [r.tflops for r in recs]
        mean_map[k] = (mean(lats), mean(tfs))
        median_map[k] = (median(lats), median(tfs))
    return mean_map, median_map


def compute_mamba_stats(per_gpu: Dict[str, Dict[int, MambaRecord]]) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Tuple[float, float]]]:
    by_seq: Dict[int, List[MambaRecord]] = {}
    for gpu_map in per_gpu.values():
        for s, rec in gpu_map.items():
            by_seq.setdefault(s, []).append(rec)
    mean_map: Dict[int, Tuple[float, float]] = {}
    median_map: Dict[int, Tuple[float, float]] = {}
    for s, recs in by_seq.items():
        lats = [r.latency_ms for r in recs]
        tfs = [r.tflops for r in recs]
        mean_map[s] = (mean(lats), mean(tfs))
        median_map[s] = (median(lats), median(tfs))
    return mean_map, median_map


def print_matmul_stats(title_suffix: str, stats_map: Dict[int, Tuple[float, float]]) -> None:
    if not stats_map:
        return
    print(f"## FP16 Matmul Benchmark (8192×8192) - {title_suffix}")
    print()
    print("| K     | Latency (s) | Throughput (TFLOPs) |")
    print("|-------|-------------|---------------------|")
    for k in sorted(stats_map.keys()):
        lat, tf = stats_map[k]
        throughput_int = int(round(tf * 1000))
        print(f"| {k:5d} | {lat:0.6f}    | {throughput_int:<19d} |")
    print()


def print_mamba_stats(title_suffix: str, stats_map: Dict[int, Tuple[float, float]]) -> None:
    if not stats_map:
        return
    print(f"## Mamba2_chunk_scan Benchmark - {title_suffix}")
    print()
    print("| Seq_len| Latency (ms) | Throughput (TFLOPs) |")
    print("|-------|-------------|---------------------|")
    for s in sorted(stats_map.keys()):
        lat, tf = stats_map[s]
        print(f"| {s:5d} | {lat:0.3f}    | {tf:0.3f}                 |")
    print()


def collect(results_root: Path) -> Tuple[Dict[int, MatmulRecord], Dict[int, MatmulRecord], Dict[int, MambaRecord]]:
    matmul_logs = list(results_root.rglob("gpu*/matmul/benchmark.log"))
    fp8_logs = list(results_root.rglob("gpu*/matmul_fp8/benchmark.log"))
    mamba_logs = list(results_root.rglob("gpu*/mamba2/benchmark.log"))

    matmul_per_gpu = [parse_matmul_log(p) for p in matmul_logs]
    fp8_per_gpu = [parse_matmul_log(p) for p in fp8_logs]
    mamba_per_gpu = [parse_mamba_log(p) for p in mamba_logs]

    best_matmul = aggregate_best_matmul(matmul_per_gpu)
    best_fp8 = aggregate_best_matmul(fp8_per_gpu)
    best_mamba = aggregate_best_mamba(mamba_per_gpu)
    return best_matmul, best_fp8, best_mamba


def collect_per_gpu(results_root: Path) -> Tuple[Dict[str, Dict[int, MatmulRecord]], Dict[str, Dict[int, MatmulRecord]], Dict[str, Dict[int, MambaRecord]]]:
    gpu_dirs = [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith("gpu")]
    per_gpu_matmul: Dict[str, Dict[int, MatmulRecord]] = {}
    per_gpu_fp8: Dict[str, Dict[int, MatmulRecord]] = {}
    per_gpu_mamba: Dict[str, Dict[int, MambaRecord]] = {}
    for g in gpu_dirs:
        gpu_name = g.name
        mm_log = g / "matmul" / "benchmark.log"
        fp8_log = g / "matmul_fp8" / "benchmark.log"
        mb_log = g / "mamba2" / "benchmark.log"
        if mm_log.exists():
            per_gpu_matmul[gpu_name] = parse_matmul_log(mm_log)
        if fp8_log.exists():
            per_gpu_fp8[gpu_name] = parse_matmul_log(fp8_log)
        if mb_log.exists():
            per_gpu_mamba[gpu_name] = parse_mamba_log(mb_log)
    return per_gpu_matmul, per_gpu_fp8, per_gpu_mamba


def gpu_sort_key(g: str) -> Tuple[int, str]:
    try:
        return (int(g[3:]), g)
    except Exception:
        return (1 << 30, g)


def print_per_gpu_sections(per_gpu_matmul: Dict[str, Dict[int, MatmulRecord]], per_gpu_fp8: Dict[str, Dict[int, MatmulRecord]], per_gpu_mamba: Dict[str, Dict[int, MambaRecord]]) -> None:
    gpu_names = sorted(set(per_gpu_matmul.keys()) | set(per_gpu_fp8.keys()) | set(per_gpu_mamba.keys()), key=gpu_sort_key)
    for g in gpu_names:
        print(f"### {g}")
        print()
        if g in per_gpu_matmul and per_gpu_matmul[g]:
            print_matmul_table("FP16 Matmul Benchmark (8192×8192)", per_gpu_matmul[g])
        if g in per_gpu_fp8 and per_gpu_fp8[g]:
            print_matmul_table("FP8 Matmul Benchmark (8192×8192)", per_gpu_fp8[g])
        if g in per_gpu_mamba and per_gpu_mamba[g]:
            print_mamba_table("Mamba2_chunk_scan Benchmark", per_gpu_mamba[g])


def update_readmes(root: Path, best_matmul: Dict[int, MatmulRecord], best_fp8: Dict[int, MatmulRecord], best_mamba: Dict[int, MambaRecord]) -> None:
    if best_matmul:
        replace_results_table(root / "matmul" / "README.md", matmul_table_lines(best_matmul))
    if best_fp8:
        replace_results_table(root / "matmul_fp8" / "README.md", matmul_table_lines(best_fp8))
    if best_mamba:
        replace_results_table(root / "mamba2" / "README.md", mamba_table_lines(best_mamba))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark results into README-style tables")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent, help="Benchmark root directory (default: scripts/..)")
    parser.add_argument("--write-readme", action="store_true", help="Update README.md tables in-place under each benchmark directory")
    args = parser.parse_args()

    roots = find_results_roots(args.root)
    if not roots:
        print("No results directories found.")
        return

    all_best_matmul: Dict[int, MatmulRecord] = {}
    all_best_fp8: Dict[int, MatmulRecord] = {}
    all_best_mamba: Dict[int, MambaRecord] = {}

    per_gpu_all_matmul: Dict[str, Dict[int, MatmulRecord]] = {}
    per_gpu_all_fp8: Dict[str, Dict[int, MatmulRecord]] = {}
    per_gpu_all_mamba: Dict[str, Dict[int, MambaRecord]] = {}

    for r in roots:
        best_matmul, best_fp8, best_mamba = collect(r)
        for k, rec in best_matmul.items():
            if k not in all_best_matmul or rec.tflops > all_best_matmul[k].tflops:
                all_best_matmul[k] = rec
        for k, rec in best_fp8.items():
            if k not in all_best_fp8 or rec.tflops > all_best_fp8[k].tflops:
                all_best_fp8[k] = rec
        for s, rec in best_mamba.items():
            if s not in all_best_mamba or rec.tflops > all_best_mamba[s].tflops:
                all_best_mamba[s] = rec

        mm_map, fp8_map, mb_map = collect_per_gpu(r)
        for g, m in mm_map.items():
            per_gpu_all_matmul[g] = m
        for g, m in fp8_map.items():
            per_gpu_all_fp8[g] = m
        for g, m in mb_map.items():
            per_gpu_all_mamba[g] = m

    # Overall best-of summaries
    if all_best_matmul:
        print_matmul_table("FP16 Matmul Benchmark (8192×8192)", all_best_matmul)
    if all_best_fp8:
        print_matmul_table("FP8 Matmul Benchmark (8192×8192)", all_best_fp8)
    if all_best_mamba:
        print_mamba_table("Mamba2_chunk_scan Benchmark", all_best_mamba)

    # Mean/Median across GPUs
    mean_mm, median_mm = compute_matmul_stats(per_gpu_all_matmul)
    mean_fp8, median_fp8 = compute_matmul_stats(per_gpu_all_fp8)
    mean_mb, median_mb = compute_mamba_stats(per_gpu_all_mamba)

    if mean_mm:
        print_matmul_stats("Mean across GPUs", mean_mm)
    if median_mm:
        print_matmul_stats("Median across GPUs", median_mm)
    if mean_fp8:
        # reuse matmul-style for FP8 table
        print("## FP8 Matmul Benchmark (8192×8192) - Mean across GPUs")
        print()
        print("| K     | Latency (s) | Throughput (TFLOPs) |")
        print("|-------|-------------|---------------------|")
        for k in sorted(mean_fp8.keys()):
            lat, tf = mean_fp8[k]
            throughput_int = int(round(tf * 1000))
            print(f"| {k:5d} | {lat:0.6f}    | {throughput_int:<19d} |")
        print()
    if median_fp8:
        print("## FP8 Matmul Benchmark (8192×8192) - Median across GPUs")
        print()
        print("| K     | Latency (s) | Throughput (TFLOPs) |")
        print("|-------|-------------|---------------------|")
        for k in sorted(median_fp8.keys()):
            lat, tf = median_fp8[k]
            throughput_int = int(round(tf * 1000))
            print(f"| {k:5d} | {lat:0.6f}    | {throughput_int:<19d} |")
        print()
    if mean_mb:
        print_mamba_stats("Mean across GPUs", mean_mb)
    if median_mb:
        print_mamba_stats("Median across GPUs", median_mb)

    # Per-GPU sections
    print_per_gpu_sections(per_gpu_all_matmul, per_gpu_all_fp8, per_gpu_all_mamba)

    if args.write_readme:
        update_readmes(args.root, all_best_matmul, all_best_fp8, all_best_mamba)


if __name__ == "__main__":
    main()
