#!/usr/bin/env python3
"""Auto-detect NPU environment and generate NPU configuration file.

Usage:
    python -m server.npu.tools.npu_detect
    python -m server.npu.tools.npu_detect --output custom_npu.yml
    python -m server.npu.tools.npu_detect --max-tasks 4 --mode shared
"""

from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_hostname() -> str | None:
    try:
        return socket.gethostname()
    except Exception:
        return None


def run_npu_smi_info() -> str | None:
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except FileNotFoundError:
        print("npu-smi not found. Please ensure Ascend/CANN runtime is installed.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running npu-smi info: {e}")
        return None


def detect_npus_npu_smi() -> list[dict] | None:
    output = run_npu_smi_info()
    if output is None:
        return None

    # Example first line: | 0     910B1               | OK  | ...
    head_line_re = re.compile(r"^\|\s*(\d+)\s+([A-Za-z0-9._-]+)\s+\|")
    bus_id_re = re.compile(r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9]")
    fraction_re = re.compile(r"(\d+)\s*/\s*(\d+)")

    npus: list[dict] = []
    current: dict | None = None

    for raw_line in output.splitlines():
        line = raw_line.rstrip()

        # Device first row
        if "No running processes found in NPU" in line:
            break
        m_head = head_line_re.match(line)
        if m_head and "NPU" not in line and "Version" not in line:
            npu_id = int(m_head.group(1))
            name = m_head.group(2).strip()
            current = {
                "logical_id": npu_id,
                "npu_smi_id": npu_id,
                "visible_id": npu_id,
                "name": name,
                "uuid": None,
                "memory_gb": None,
                "bus_id": None,
            }
            continue

        # Device second row (contains bus id + HBM usage fraction)
        if current is not None and line.lstrip().startswith("|") and bus_id_re.search(line):
            bus_id = bus_id_re.search(line).group(0)
            pairs = fraction_re.findall(line)
            total_mb = int(pairs[-1][1]) if pairs else 0
            current["bus_id"] = bus_id
            current["memory_gb"] = round(total_mb / 1024.0, 2) if total_mb > 0 else None
            npus.append(current)
            current = None

    # Filter malformed entries
    npus = [n for n in npus if isinstance(n.get("logical_id"), int)]
    # Deduplicate by logical id (keep first complete entry)
    unique_by_id: dict[int, dict] = {}
    for npu in npus:
        npu_id = npu["logical_id"]
        if npu_id not in unique_by_id:
            unique_by_id[npu_id] = npu
    npus = [unique_by_id[k] for k in sorted(unique_by_id.keys())]
    return npus


def generate_yaml_config(
    npus: list[dict],
    default_mode: str = "shared",
    memory_threshold: float = 1.0,
    max_concurrent_tasks: int = 2,
    hostname: str | None = None,
) -> str:
    lines = [
        "# Auto-generated NPU configuration",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Total NPUs: {len(npus)}",
    ]
    if hostname:
        lines.append(f"# Hostname: {hostname}")
    lines.append("")
    lines.append("npus:")

    for npu in npus:
        lines.extend(
            [
                f"  - logical_id: {npu['logical_id']}",
                f"    npu_smi_id: {npu['npu_smi_id']}",
                f"    visible_id: {npu['visible_id']}",
                f"    name: \"{npu['name']}\"",
                f"    uuid: \"{npu.get('uuid') or ''}\"",
                f"    memory_gb: {npu.get('memory_gb') if npu.get('memory_gb') is not None else 0}",
                "    enabled: true",
                f"    default_mode: \"{default_mode}\"",
                f"    memory_threshold: {memory_threshold}",
                f"    max_concurrent_tasks: {max_concurrent_tasks}",
                "",
            ]
        )

    lines.extend(
        [
            "server:",
            "  auto_register_npus: true",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Auto-detect NPUs and generate NPU configuration file")
    parser.add_argument("--output", "-o", help="Output config file path (default: hostname-based filename)")
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Base directory for config files (default: server/npu/configs/npu_resources/)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["shared", "exclusive"],
        default="shared",
        help="Default NPU mode",
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=1.0,
        help="Memory threshold (0.0-1.0, default: 1.0)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=2,
        help="Max concurrent tasks per NPU (default: 2)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config to stdout instead of writing file")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the npu-server startup command after generating config",
    )
    args = parser.parse_args()

    npus = detect_npus_npu_smi()
    if npus is None:
        sys.exit(1)
    if not npus:
        print("No NPUs detected. Exiting.")
        sys.exit(1)

    hostname = get_hostname()
    print(f"Detected {len(npus)} NPU(s):")
    for npu in npus:
        print(
            f"  NPU {npu['logical_id']}: {npu['name']} "
            f"(Bus-Id: {npu.get('bus_id')}, memory_gb: {npu.get('memory_gb')})"
        )

    yaml_content = generate_yaml_config(
        npus=npus,
        default_mode=args.mode,
        memory_threshold=args.memory_threshold,
        max_concurrent_tasks=args.max_tasks,
        hostname=hostname,
    )

    if args.dry_run:
        print("\n" + "=" * 60)
        print("Generated Configuration:")
        print("=" * 60)
        print(yaml_content)
        if args.print_command:
            print("\n" + "=" * 60)
            print("NPU Server Startup Command:")
            print("=" * 60)
            print("npu-server --npu-config <config_file_path>")
            print("=" * 60)
        return

    if args.output:
        output_path = Path(args.output)
    else:
        config_name = f"{hostname or 'local'}_npu_config.yml"
        config_dir = Path(args.config_dir) if args.config_dir else Path("server/npu/configs/npu_resources")
        output_path = config_dir / config_name

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\nConfiguration written to: {output_path}")

    if args.print_command:
        print("\n" + "=" * 60)
        print("NPU Server Startup Command:")
        print("=" * 60)
        print(f"npu-server --npu-config {output_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()

