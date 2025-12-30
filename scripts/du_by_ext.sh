#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  du_by_ext.sh [DIR] [TOP_N]

Description:
  Summarize total file sizes by "type" (file extension, including compound extensions
  like tar.gz), and print the largest TOP_N files.

Examples:
  bash scripts/du_by_ext.sh .
  bash scripts/du_by_ext.sh /path/to/dir 50
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DIR="${1:-.}"
TOP_N="${2:-20}"

python3 - "$DIR" "$TOP_N" <<'PY'
import os
import sys
import heapq
from collections import defaultdict

def human_size(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(f)} {u}"
            return f"{f:.2f} {u}"
        f /= 1024.0
    return f"{f:.2f} PiB"

COMPOUND_EXTS = (
    "tar.gz",
    "tar.bz2",
    "tar.xz",
    "tar.zst",
    "tar.lz4",
    "tar.lz",
    "tar",
)

def ext_of(name: str) -> str:
    lower = name.lower()
    for e in COMPOUND_EXTS:
        if lower.endswith("." + e):
            return e
    if lower.startswith(".") and lower.count(".") == 1:
        return "(dotfile)"
    # Split on last dot only
    root, dot, ext = lower.rpartition(".")
    if dot == "" or ext == "":
        return "(noext)"
    return ext

def walk_files(root: str):
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            yield entry
                    except OSError:
                        continue
        except OSError:
            continue

def main() -> int:
    if len(sys.argv) < 2:
        print("ERROR: missing DIR argument", file=sys.stderr)
        return 2
    root = sys.argv[1]
    try:
        top_n = int(sys.argv[2]) if len(sys.argv) >= 3 else 20
    except ValueError:
        print("ERROR: TOP_N must be an integer", file=sys.stderr)
        return 2
    if top_n < 0:
        print("ERROR: TOP_N must be >= 0", file=sys.stderr)
        return 2

    sums = defaultdict(int)
    counts = defaultdict(int)
    total = 0

    # Min-heap of (size, path), keeping only top_n entries
    heap = []

    for entry in walk_files(root):
        try:
            st = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        size = int(st.st_size)
        total += size
        e = ext_of(entry.name)
        sums[e] += size
        counts[e] += 1

        if top_n > 0:
            item = (size, entry.path)
            if len(heap) < top_n:
                heapq.heappush(heap, item)
            else:
                if size > heap[0][0]:
                    heapq.heapreplace(heap, item)

    print(f"Directory: {root}")
    print(f"Total: {human_size(total)} ({total} bytes)")
    print("")
    print("By extension (sorted by total bytes):")
    rows = sorted(sums.items(), key=lambda kv: kv[1], reverse=True)
    if not rows:
        print("  (no files)")
    else:
        print(f"{'ext':<16}  {'files':>10}  {'bytes':>16}  {'human':>12}")
        for e, b in rows:
            c = counts[e]
            print(f"{e:<16}  {c:>10}  {b:>16}  {human_size(b):>12}")

    if top_n > 0:
        print("")
        print(f"Top {top_n} largest files:")
        for size, path in sorted(heap, key=lambda x: x[0], reverse=True):
            print(f"{human_size(size):>12}  {size:>16}  {path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
PY


