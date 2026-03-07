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


# (serve) ubuntu@g0013:/data/apps/project/cu2tri$ bash /data/apps/project/cu2tri/scripts/du_by_ext.sh /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/
# Directory: /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/
# Total: 3.79 GiB (4066144566 bytes)

# By extension (sorted by total bytes):
# ext                    files             bytes         human
# so                      2711        2685156000      2.50 GiB
# jsonl                   5670         807190012    769.80 MiB
# json                   10841         350804253    334.55 MiB
# log                    15698         117748966    112.29 MiB
# py                     27889          77720327     74.12 MiB
# pyc                     8469          24107845     22.99 MiB
# cu                      2845           3414295      3.26 MiB
# txt                        2              2868      2.80 KiB
# 22783238934640             1                 0           0 B

# Top 20 largest files:
#    12.20 MiB          12797031  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/gpt_5_mini_xpiler.log
#    10.42 MiB          10928631  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251023_014909/gpt_5_mini_xpiler.log
#     4.39 MiB           4602435  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/deformable_4_8_256_200_4_4/attempt_04/logs/retry_events.jsonl
#     4.39 MiB           4598689  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/gpt_5_mini_xpiler.jsonl
#     3.89 MiB           4080226  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251023_014909/gpt_5_mini_xpiler.jsonl
#     3.46 MiB           3622900  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/depthwiseconv_192_3_128/attempt_04/logs/retry_events.jsonl
#     3.44 MiB           3610357  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/mha_1_4096_12_512/attempt_02/logs/retry_events.jsonl
#     3.29 MiB           3444924  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/scatter_1_512_7_7/attempt_01/logs/retry_events.jsonl
#     3.27 MiB           3433260  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/gqa_2_2_16_16_512/attempt_04/logs/retry_events.jsonl
#     3.19 MiB           3341737  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/mha_1_2048_6_256/attempt_05/logs/retry_events.jsonl
#     3.16 MiB           3313413  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/mha_1_2048_12_256/attempt_04/logs/retry_events.jsonl
#     2.75 MiB           2887939  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/avgpool_4_35_35_192_5_5_2_2/attempt_01/logs/retry_events.jsonl
#     2.69 MiB           2820951  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/gqa_1_4_16_16_512/attempt_05/logs/retry_events.jsonl
#     2.55 MiB           2677479  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/deformable_4_8_256_200_4_4/attempt_04/logs/conversations/all_conversations.jsonl
#     2.55 MiB           2672515  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/maxpool_5_32_32_64_5_5_3_3/attempt_05/logs/retry_events.jsonl
#     2.53 MiB           2651515  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/avgpool_16_112_112_64_5_5_3_3/attempt_03/logs/retry_events.jsonl
#     2.47 MiB           2590347  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/layernorm_1_4_128/attempt_01/logs/retry_events.jsonl
#     2.46 MiB           2575239  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/maxpool_5_32_32_64_5_5_3_3/attempt_04/logs/retry_events.jsonl
#     2.37 MiB           2483160  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/maxpool_4_56_56_128_5_5_2_2/attempt_03/logs/retry_events.jsonl
#     2.33 MiB           2443714  /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/maxpool_4_35_35_192_5_5_3_3/attempt_01/logs/retry_events.jsonl