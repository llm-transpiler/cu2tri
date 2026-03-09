#!/usr/bin/env python3
"""
Plot round-distribution chart from xpiler batch log.

Usage:
  python llm_trans/scripts/plot_success_round_distribution.py \
      --log /workspace/llm_trans/runs/cu2asc/xpiler/gpt_5/20260310_023336/gpt_5_xpiler.log \
      --out-png /workspace/llm_trans/runs/cu2asc/xpiler/gpt_5/20260310_023336/success_round_distribution.png \
      --out-csv /workspace/llm_trans/runs/cu2asc/xpiler/gpt_5/20260310_023336/success_round_distribution.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# Example: "✅ add             (8/8)" / "❌ avgpool         (0/8)"
RE_CASE_TYPE = re.compile(r"^\s*[-0-9:, ]+\s*-\s*INFO\s*-\s*[✅❌]\s+([A-Za-z0-9_]+)\s+\((\d+)/(\d+)\)\s*$")
# Example: "✅ (Attempt 1, Round 1) add_64"
RE_CASE_SUCCESS = re.compile(
    r"^\s*[-0-9:, ]+\s*-\s*INFO\s*-\s*✅\s+\(Attempt\s+(\d+),\s*Round\s+(\d+)\)\s+([A-Za-z0-9_]+)\s*$"
)
# Example: "❌ (Attempts 5) avgpool_xxx"
RE_CASE_FAIL = re.compile(
    r"^\s*[-0-9:, ]+\s*-\s*INFO\s*-\s*❌\s+\(Attempts\s+(\d+)\)\s+([A-Za-z0-9_]+)\s*$"
)


@dataclass
class CaseRecord:
    case_name: str
    success: bool
    attempt: Optional[int] = None
    round_id: Optional[int] = None
    fail_attempts: Optional[int] = None


@dataclass
class CaseTypeStats:
    case_type: str
    passed: int = 0
    total: int = 0
    cases: List[CaseRecord] = field(default_factory=list)

    def success_rate(self) -> float:
        return (self.passed / self.total * 100.0) if self.total else 0.0

    def round_distribution(self, rounds: Tuple[int, ...] = (1, 2, 3, 4, 5)) -> Dict[int, float]:
        """
        Distribution is over all cases in this case_type (not only passed cases),
        so each row always sums to <= 100 and the remaining part can be shown as failed.
        """
        dist = {r: 0.0 for r in rounds}
        if self.total == 0:
            return dist

        for c in self.cases:
            if c.success and c.round_id in dist:
                dist[c.round_id] += 100.0 / self.total
        return dist

    def fail_percentage(self) -> float:
        return max(0.0, 100.0 - sum(self.round_distribution().values()))

    def mean_success_round(self) -> Optional[float]:
        rounds = [c.round_id for c in self.cases if c.success and c.round_id is not None]
        if not rounds:
            return None
        return sum(rounds) / len(rounds)

    def mean_success_attempt(self) -> Optional[float]:
        attempts = [c.attempt for c in self.cases if c.success and c.attempt is not None]
        if not attempts:
            return None
        return sum(attempts) / len(attempts)


def parse_log(log_path: str) -> Dict[str, CaseTypeStats]:
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_summary = False
    current_case_type: Optional[str] = None
    stats: Dict[str, CaseTypeStats] = {}

    for line in lines:
        if "DETAILED BATCH TESTING SUMMARY" in line:
            in_summary = True
            continue
        if in_summary and "OVERALL TOTAL" in line:
            break
        if not in_summary:
            continue

        m_type = RE_CASE_TYPE.match(line)
        if m_type:
            case_type, passed, total = m_type.group(1), int(m_type.group(2)), int(m_type.group(3))
            current_case_type = case_type
            stats[case_type] = CaseTypeStats(case_type=case_type, passed=passed, total=total)
            continue

        if current_case_type is None:
            continue

        m_ok = RE_CASE_SUCCESS.match(line)
        if m_ok:
            attempt = int(m_ok.group(1))
            round_id = int(m_ok.group(2))
            case_name = m_ok.group(3)
            stats[current_case_type].cases.append(
                CaseRecord(
                    case_name=case_name,
                    success=True,
                    attempt=attempt,
                    round_id=round_id,
                )
            )
            continue

        m_fail = RE_CASE_FAIL.match(line)
        if m_fail:
            attempts = int(m_fail.group(1))
            case_name = m_fail.group(2)
            stats[current_case_type].cases.append(
                CaseRecord(
                    case_name=case_name,
                    success=False,
                    fail_attempts=attempts,
                )
            )
            continue

    return stats


def save_csv(stats: Dict[str, CaseTypeStats], out_csv: str) -> None:
    rows = []
    for case_type, st in sorted(stats.items()):
        dist = st.round_distribution()
        rows.append(
            {
                "case_type": case_type,
                "passed": st.passed,
                "total": st.total,
                "success_rate_pct": f"{st.success_rate():.2f}",
                "round1_pct": f"{dist[1]:.2f}",
                "round2_pct": f"{dist[2]:.2f}",
                "round3_pct": f"{dist[3]:.2f}",
                "round4_pct": f"{dist[4]:.2f}",
                "round5_pct": f"{dist[5]:.2f}",
                "fail_pct": f"{st.fail_percentage():.2f}",
                "mean_success_round": "" if st.mean_success_round() is None else f"{st.mean_success_round():.2f}",
                "mean_success_attempt": "" if st.mean_success_attempt() is None else f"{st.mean_success_attempt():.2f}",
            }
        )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def save_case_detail_csv(stats: Dict[str, CaseTypeStats], out_csv: str) -> None:
    rows = []
    for case_type, st in sorted(stats.items()):
        for c in st.cases:
            rows.append(
                {
                    "case_type": case_type,
                    "case_name": c.case_name,
                    "success": int(c.success),
                    "attempt": "" if c.attempt is None else c.attempt,
                    "round_id": "" if c.round_id is None else c.round_id,
                    "fail_attempts": "" if c.fail_attempts is None else c.fail_attempts,
                }
            )
    if not rows:
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_distribution(
    stats: Dict[str, CaseTypeStats],
    out_png: str,
    title: str,
    include_fail: bool = True,
    sort_by: str = "name",
    only_success_types: bool = False,
) -> None:
    if not stats:
        raise RuntimeError("No case stats parsed from log.")

    items = list(stats.items())
    if only_success_types:
        items = [(k, v) for k, v in items if v.passed > 0]
        if not items:
            raise RuntimeError("No case types with passed > 0 after filtering.")

    if sort_by == "success_rate":
        items.sort(key=lambda x: x[1].success_rate(), reverse=True)
    else:
        items.sort(key=lambda x: x[0])

    case_types = [k for k, _ in items]
    round_keys = [1, 2, 3, 4, 5]
    round_colors = {
        1: "#7fbf7b",
        2: "#b8d98a",
        3: "#e8e2a4",
        4: "#f1c27d",
        5: "#e57f6e",
    }
    fail_color = "#c7c7c7"

    fig, ax = plt.subplots(figsize=(14, max(6, 0.42 * len(case_types))))
    y_pos = list(range(len(case_types)))
    left = [0.0] * len(case_types)

    for r in round_keys:
        values = [st.round_distribution()[r] for _, st in items]
        bars = ax.barh(
            y_pos,
            values,
            left=left,
            color=round_colors[r],
            edgecolor="white",
            linewidth=0.6,
            label=f"Round {r}",
        )
        # Label only visible segments.
        for bar, v in zip(bars, values):
            if v >= 7.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"R{r}: {v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )
        left = [l + v for l, v in zip(left, values)]

    if include_fail:
        fail_vals = [st.fail_percentage() for _, st in items]
        ax.barh(
            y_pos,
            fail_vals,
            left=left,
            color=fail_color,
            edgecolor="white",
            linewidth=0.6,
            label="Failed",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(case_types)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of Cases")
    ax.set_ylabel("Case Type")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def print_markdown_table(stats: Dict[str, CaseTypeStats], top_n: int = 36) -> None:
    items = sorted(stats.items(), key=lambda x: x[1].success_rate())
    print("\n| case_type | pass/total | success_rate | R1 | R2 | R3 | R4 | R5 | fail |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case_type, st in items[:top_n]:
        dist = st.round_distribution()
        print(
            f"| {case_type} | {st.passed}/{st.total} | {st.success_rate():.1f}% "
            f"| {dist[1]:.1f}% | {dist[2]:.1f}% | {dist[3]:.1f}% | {dist[4]:.1f}% | {dist[5]:.1f}% "
            f"| {st.fail_percentage():.1f}% |"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot success round distribution by case type from xpiler log.")
    p.add_argument("--log", required=True, help="Path to gpt_5_xpiler.log")
    p.add_argument("--out-png", required=True, help="Path to output PNG")
    p.add_argument("--out-csv", required=True, help="Path to output CSV summary")
    p.add_argument(
        "--out-case-csv",
        default="",
        help="Optional path to output per-case detail CSV.",
    )
    p.add_argument("--title", default="Success Round Distribution by Case Type - gpt_5", help="Plot title")
    p.add_argument(
        "--sort-by",
        choices=("name", "success_rate"),
        default="name",
        help="Sort y-axis by case type name or success rate.",
    )
    p.add_argument(
        "--hide-fail",
        action="store_true",
        help="Hide failed percentage segment in chart.",
    )
    p.add_argument(
        "--only-success-types",
        action="store_true",
        help="Only include case types with passed > 0 in the plot.",
    )
    p.add_argument(
        "--print-table",
        action="store_true",
        help="Print markdown table to stdout.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    stats = parse_log(args.log)
    save_csv(stats, args.out_csv)
    if args.out_case_csv:
        save_case_detail_csv(stats, args.out_case_csv)
    plot_distribution(
        stats=stats,
        out_png=args.out_png,
        title=args.title,
        include_fail=not args.hide_fail,
        sort_by=args.sort_by,
        only_success_types=args.only_success_types,
    )

    total_cases = sum(st.total for st in stats.values())
    total_pass = sum(st.passed for st in stats.values())
    print(f"Parsed case types: {len(stats)}")
    print(f"Overall: {total_pass}/{total_cases} ({(total_pass / total_cases * 100.0) if total_cases else 0.0:.2f}%)")
    print(f"PNG: {args.out_png}")
    print(f"CSV: {args.out_csv}")
    if args.out_case_csv:
        print(f"Case CSV: {args.out_case_csv}")
    if args.print_table:
        print_markdown_table(stats)


if __name__ == "__main__":
    main()
