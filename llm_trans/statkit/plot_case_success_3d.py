#!/usr/bin/env python3

"""
Plot 3D surfaces from xpiler case_success.json statistics.

For each JSON file, we aggregate across all ops and cases and build a surface:
  - X axis (attempts): threshold on attempt_number (<= x)
  - Y axis (rounds): threshold on round_final (<= y)
  - Z axis: number of cases (or success rate) that have at least one successful
            attempt with attempt_number <= x and round_final <= y.

One figure is generated per JSON file.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def load_case_success(json_path: Path) -> Dict[str, Any]:
    """Load a case_success.json file."""
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_max_attempts_rounds(data: Dict[str, Any]) -> Tuple[int, int]:
    """Infer global max_attempts and max_rounds from the JSON structure."""
    max_attempt = 0
    max_round = 0

    for op_name, op_cases in data.items():
        if not isinstance(op_cases, dict):
            continue
        for case_name, stats in op_cases.items():
            attempts = stats.get("attempts", [])
            for rec in attempts:
                a = rec.get("attempt_number")
                r = rec.get("round_final")
                if isinstance(a, int) and a > max_attempt:
                    max_attempt = a
                if isinstance(r, int) and r > max_round:
                    max_round = r

    if max_attempt <= 0 or max_round <= 0:
        raise ValueError("No attempts/rounds found in JSON data.")

    return max_attempt, max_round


def build_success_surfaces(
    data: Dict[str, Any],
    max_attempts: int | None = None,
    max_rounds: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Build 3D surfaces:
      - count_surface[ry, ax]: number of successful cases
      - rate_surface[ry, ax]: success rate in [0, 1]

    Indices are:
      ax -> attempts threshold index (0-based, corresponds to attempts = ax + 1)
      ry -> rounds threshold index (0-based, corresponds to rounds = ry + 1)
    """
    inferred_max_attempts, inferred_max_rounds = infer_max_attempts_rounds(data)

    if max_attempts is None:
        max_attempts = inferred_max_attempts
    else:
        max_attempts = min(max_attempts, inferred_max_attempts)

    if max_rounds is None:
        max_rounds = inferred_max_rounds
    else:
        max_rounds = min(max_rounds, inferred_max_rounds)

    if max_attempts <= 0 or max_rounds <= 0:
        raise ValueError("max_attempts and max_rounds must be positive.")

    # Collect per-case success patterns
    case_masks: List[np.ndarray] = []

    for op_name, op_cases in data.items():
        if not isinstance(op_cases, dict):
            continue
        for case_name, stats in op_cases.items():
            attempts = stats.get("attempts", [])
            if not attempts:
                continue

            # Boolean mask for this case: [round, attempt]
            mask = np.zeros((max_rounds, max_attempts), dtype=bool)

            for rec in attempts:
                if not rec.get("success"):
                    continue
                a = rec.get("attempt_number")
                r = rec.get("round_final")
                if not isinstance(a, int) or not isinstance(r, int):
                    continue
                if a < 1 or r < 1:
                    continue
                if a > max_attempts or r > max_rounds:
                    # Ignore successes outside the capped region
                    continue

                # Success at (a, r) implies success for all thresholds
                # attempts >= a and rounds >= r.
                a_idx = a - 1
                r_idx = r - 1
                mask[r_idx:, a_idx:] = True

            if mask.any():
                case_masks.append(mask)

    if not case_masks:
        raise ValueError("No successful cases found in JSON data.")

    # Stack and aggregate
    stack = np.stack(case_masks, axis=0)  # [num_cases, rounds, attempts]
    count_surface = stack.sum(axis=0).astype(float)
    num_cases = stack.shape[0]
    rate_surface = count_surface / float(num_cases)

    return count_surface, rate_surface, max_attempts, max_rounds


def plot_surface_3d(
    count_surface: np.ndarray,
    rate_surface: np.ndarray,
    max_attempts: int,
    max_rounds: int,
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot a 3D surface and save to file."""
    attempts = np.arange(1, max_attempts + 1)
    rounds = np.arange(1, max_rounds + 1)
    A, R = np.meshgrid(attempts, rounds)  # shapes: [rounds, attempts]

    if metric == "count":
        Z = count_surface
        z_label = "Successful cases"
    elif metric == "rate":
        Z = rate_surface
        z_label = "Success rate"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(A, R, Z, cmap=cm.viridis, edgecolor="none", antialiased=True)

    ax.set_xlabel("Attempts threshold (<= x)")
    ax.set_ylabel("Rounds threshold (<= y)")
    ax.set_zlabel(z_label)
    ax.set_title(title)

    # Flip attempts axis so that attempts=1 is on the right, max_attempts on the left.
    # Visually this makes the success surface slope from bottom-right to top-left.
    ax.set_xlim(max_attempts, 1)

    cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
    cbar.set_label(z_label)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def derive_title_and_output(json_path: Path, metric: str) -> Tuple[str, Path]:
    """Generate a human-readable title and default output path for a JSON file."""
    # Expect path like .../stats/cu2tri/xpiler/<model>/<timestamp>/case_success.json
    parts = json_path.parts
    title = json_path.stem

    # Try to include model and timestamp if present
    # Look for pattern ".../xpiler/<model>/<timestamp>/case_success.json"
    try:
        xpiler_idx = parts.index("xpiler")
        model = parts[xpiler_idx + 1]
        timestamp = parts[xpiler_idx + 2]
        title = f"{model} {timestamp} ({metric})"
    except ValueError:
        # "xpiler" not in path; keep default
        pass
    except IndexError:
        # Not enough components after "xpiler"
        pass

    # Default output in the same directory
    out_name = f"{json_path.stem}_3d_{metric}.png"
    output_path = json_path.with_name(out_name)
    return title, output_path


def process_one_file(
    json_path: Path,
    metric: str,
    max_attempts: int | None,
    max_rounds: int | None,
    output_dir: Path | None,
) -> None:
    data = load_case_success(json_path)
    count_surface, rate_surface, ma, mr = build_success_surfaces(
        data, max_attempts=max_attempts, max_rounds=max_rounds
    )

    title, default_output = derive_title_and_output(json_path, metric)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_name = default_output.name
        output_path = output_dir / out_name
    else:
        output_path = default_output

    print(f"[plot] {json_path} -> {output_path} (metric={metric}, attempts<= {ma}, rounds<= {mr})")

    plot_surface_3d(
        count_surface=count_surface,
        rate_surface=rate_surface,
        max_attempts=ma,
        max_rounds=mr,
        metric=metric,
        title=title,
        output_path=output_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 3D surfaces from xpiler case_success.json files.\n"
            "Z axis: successful case count or success rate for thresholds\n"
            "Attempts <= x, Rounds <= y, aggregated over all cases."
        )
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        help="Paths to case_success.json files.",
    )
    parser.add_argument(
        "--metric",
        choices=["count", "rate"],
        default="count",
        help="Z axis metric: 'count' (number of successful cases) or 'rate' (success rate).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Optional cap for attempts threshold (default: infer from data).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Optional cap for rounds threshold (default: infer from data).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for figures (default: next to each JSON file).",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir is not None else None

    for path_str in args.json_files:
        json_path = Path(path_str).expanduser()
        if not json_path.is_file():
            print(f"[warn] {json_path} is not a file, skipping.")
            continue
        try:
            process_one_file(
                json_path=json_path,
                metric=args.metric,
                max_attempts=args.max_attempts,
                max_rounds=args.max_rounds,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"[error] Failed to process {json_path}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


