#!/usr/bin/env bash
set -euo pipefail

# archive_xpiler.sh
#
# 这个脚本做什么
# --------------
# 用来把 xpiler 目录树下的各种产物“打包汇总”出来，方便从服务器传出去。
# 支持两类操作：
#
# A) 逐文件压缩（不打 tar 包）：
#   --jsonl-only
#   - 会在原地把每个 *.jsonl 生成一个同名的 *.jsonl.gz
#   - 例如：/path/to/logs/retry_events.jsonl -> /path/to/logs/retry_events.jsonl.gz
#   - 默认保留原始 .jsonl（不会删除）
#
# B) 打 tar.gz 包（会保留相对路径）：
#   1) 每个 attempt 单独一个包：
#      --triton-only
#      - 对每个 attempt_* 目录，把其下所有 triton_* 目录打成：
#        attempt_XX/triton_artifacts.tar.gz
#
#   2) 全局汇总成一个大包（相对 --root 保留路径，不会写入绝对路径）：
#      --triton-global OUT         ：收集所有 triton_* 目录
#      --conv-global OUT           ：收集所有 */attempt_*/logs/history/conversation.json
#      --retry-global OUT          ：收集所有 */attempt_*/logs/retry_events.jsonl
#      --perf-global OUT           ：收集所有 */attempt_*/logs/perf 目录（包含其下全部文件）
#      --xpiler-jsonl-global OUT   ：收集所有 *_xpiler.jsonl（通常是 model/timestamp 根目录下的汇总文件）
#
# 大文件分片
# ----------
# 默认不开分片（--split-size 0）。
# 如果开启分片（例如 --split-size 5G），当输出的 *.tar.gz 大于阈值时会生成：
#   OUT.part000, OUT.part001, ...
# 注意：脚本不会自动删除原始 OUT 文件（你可以根据需要自行删除/保留）。
#
# 推荐用法
# --------
# 1) 先用 --dry-run 预览将要执行的命令/路径
# 2) 再去掉 --dry-run 真正执行
#
# 分片合并与解压示例
# ------------------
#   cat OUT.part* > OUT
#   tar -tzf OUT | head
#   tar -xzf OUT -C /target/dir

ROOT="/data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler"
DRY_RUN=0
DO_JSONL=1
DO_TRITON=1
DELETE_AFTER=0
SPLIT_SIZE="0"
TRITON_GLOBAL_OUT=""
CONV_GLOBAL_OUT=""
RETRY_GLOBAL_OUT=""
PERF_GLOBAL_OUT=""
XPILER_JSONL_GLOBAL_OUT=""

die() { echo "ERROR: $*" >&2; exit 1; }

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

have() { command -v "$1" >/dev/null 2>&1; }

maybe_split_file() {
  local file="$1"
  local size="${SPLIT_SIZE}"

  if [[ -z "${size}" || "${size}" == "0" ]]; then
    return 0
  fi

  [[ -f "${file}" ]] || return 0
  have split || die "split is required for --split-size (coreutils)"

  # Only split when file is larger than the threshold (to avoid clutter for small outputs).
  if have numfmt; then
    local threshold_bytes
    local file_bytes
    threshold_bytes="$(numfmt --from=iec --to=none "${size}")"
    file_bytes="$(stat -c%s "${file}")"
    if [[ "${file_bytes}" -le "${threshold_bytes}" ]]; then
      return 0
    fi
  fi

  # Create parts like: foo.tar.gz.part000, foo.tar.gz.part001, ...
  run split -b "${size}" -d -a 3 -- "${file}" "${file}.part"
}

compress_jsonl() {
  local compressor="gzip"
  if have pigz; then compressor="pigz"; fi

  # Skip already compressed files (*.jsonl.gz).
  # Create foo.jsonl.gz next to foo.jsonl (keep original).
  find "${ROOT}" -type f -name '*.jsonl' ! -name '*.jsonl.gz' -print0 \
    | while IFS= read -r -d '' f; do
        if [[ -f "${f}.gz" ]]; then
          echo "skip (exists): ${f}.gz"
          continue
        fi
        # -k keep original, -f overwrite (not used here because we skip), -9 best compression.
        run "${compressor}" -9 -k -- "${f}"
      done
}

archive_triton_per_attempt() {
  local use_pigz=0
  if have pigz; then use_pigz=1; fi

  # Find attempt_* directories, then look for triton_* directories directly under each attempt.
  find "${ROOT}" -type d -name 'attempt_*' -print0 \
    | while IFS= read -r -d '' attempt_dir; do
        # Collect triton_* dirs under this attempt directory (maxdepth 1).
        mapfile -d '' triton_dirs < <(find "${attempt_dir}" -maxdepth 1 -type d -name 'triton_*' -print0)
        if [[ "${#triton_dirs[@]}" -eq 0 ]]; then
          continue
        fi

        local out="${attempt_dir}/triton_artifacts.tar.gz"
        if [[ -f "${out}" ]]; then
          echo "skip (exists): ${out}"
          continue
        fi

        # Tar wants paths relative to -C dir to avoid embedding absolute paths.
        local rels=()
        local d
        for d in "${triton_dirs[@]}"; do
          rels+=("$(basename "${d}")")
        done

        if [[ "${use_pigz}" == "1" ]]; then
          run tar -C "${attempt_dir}" -I "pigz -9" -cf "${out}" -- "${rels[@]}"
        else
          run tar -C "${attempt_dir}" -z -cf "${out}" -- "${rels[@]}"
        fi
        maybe_split_file "${out}"

        if [[ "${DELETE_AFTER}" == "1" ]]; then
          for d in "${triton_dirs[@]}"; do
            run rm -rf -- "${d}"
          done
        fi
      done
}

archive_triton_global() {
  local out="${TRITON_GLOBAL_OUT}"
  [[ -n "${out}" ]] || die "TRITON_GLOBAL_OUT is empty"
  if [[ -f "${out}" ]]; then
    echo "skip (exists): ${out}"
    return 0
  fi

  local use_pigz=0
  if have pigz; then use_pigz=1; fi

  local tmp_list
  tmp_list="$(mktemp)"
  # Cleanup explicitly to avoid trap collisions.

  # Collect triton_* directories and store their paths relative to ROOT (null-delimited for safety).
  local found=0
  while IFS= read -r -d '' d; do
    found=1
    printf '%s\0' "${d#${ROOT}/}" >>"${tmp_list}"
  done < <(find "${ROOT}" -type d -name 'triton_*' -print0)

  if [[ "${found}" -eq 0 ]]; then
    echo "no triton_* directories found under: ${ROOT}"
    rm -f "${tmp_list}"
    return 0
  fi

  run mkdir -p "$(dirname "${out}")"
  if [[ "${use_pigz}" == "1" ]]; then
    run tar -C "${ROOT}" -I "pigz -9" --null -T "${tmp_list}" -cf "${out}"
  else
    run tar -C "${ROOT}" -z --null -T "${tmp_list}" -cf "${out}"
  fi
  maybe_split_file "${out}"
  rm -f "${tmp_list}"
}

archive_conversation_history_global() {
  local out="${CONV_GLOBAL_OUT}"
  [[ -n "${out}" ]] || die "CONV_GLOBAL_OUT is empty"
  if [[ -f "${out}" ]]; then
    echo "skip (exists): ${out}"
    return 0
  fi

  local use_pigz=0
  if have pigz; then use_pigz=1; fi

  local tmp_list
  tmp_list="$(mktemp)"
  # Cleanup explicitly to avoid trap collisions.

  local found=0
  # Only include the specific history/conversation.json files, preserving relative paths under ROOT.
  while IFS= read -r -d '' f; do
    found=1
    printf '%s\0' "${f#${ROOT}/}" >>"${tmp_list}"
  done < <(find "${ROOT}" -type f -path '*/attempt_*/logs/history/conversation.json' -print0)

  if [[ "${found}" -eq 0 ]]; then
    echo "no logs/history/conversation.json files found under: ${ROOT}"
    rm -f "${tmp_list}"
    return 0
  fi

  run mkdir -p "$(dirname "${out}")"
  if [[ "${use_pigz}" == "1" ]]; then
    run tar -C "${ROOT}" -I "pigz -9" --null -T "${tmp_list}" -cf "${out}"
  else
    run tar -C "${ROOT}" -z --null -T "${tmp_list}" -cf "${out}"
  fi
  maybe_split_file "${out}"
  rm -f "${tmp_list}"
}

archive_retry_events_global() {
  local out="${RETRY_GLOBAL_OUT}"
  [[ -n "${out}" ]] || die "RETRY_GLOBAL_OUT is empty"
  if [[ -f "${out}" ]]; then
    echo "skip (exists): ${out}"
    return 0
  fi

  local use_pigz=0
  if have pigz; then use_pigz=1; fi

  local tmp_list
  tmp_list="$(mktemp)"
  # Cleanup explicitly to avoid trap collisions.

  local found=0
  # Include retry_events.jsonl files under attempts, preserving relative paths under ROOT.
  while IFS= read -r -d '' f; do
    found=1
    printf '%s\0' "${f#${ROOT}/}" >>"${tmp_list}"
  done < <(find "${ROOT}" -type f -path '*/attempt_*/logs/retry_events.jsonl' -print0)

  if [[ "${found}" -eq 0 ]]; then
    echo "no logs/retry_events.jsonl files found under: ${ROOT}"
    rm -f "${tmp_list}"
    return 0
  fi

  run mkdir -p "$(dirname "${out}")"
  if [[ "${use_pigz}" == "1" ]]; then
    run tar -C "${ROOT}" -I "pigz -9" --null -T "${tmp_list}" -cf "${out}"
  else
    run tar -C "${ROOT}" -z --null -T "${tmp_list}" -cf "${out}"
  fi
  maybe_split_file "${out}"
  rm -f "${tmp_list}"
}

archive_perf_global() {
  local out="${PERF_GLOBAL_OUT}"
  [[ -n "${out}" ]] || die "PERF_GLOBAL_OUT is empty"
  if [[ -f "${out}" ]]; then
    echo "skip (exists): ${out}"
    return 0
  fi

  local use_pigz=0
  if have pigz; then use_pigz=1; fi

  local tmp_list
  tmp_list="$(mktemp)"
  # Cleanup explicitly to avoid trap collisions.

  local found=0
  # Include logs/perf directory under attempts (archives everything under it), preserving relative paths under ROOT.
  while IFS= read -r -d '' d; do
    found=1
    printf '%s\0' "${d#${ROOT}/}" >>"${tmp_list}"
  done < <(find "${ROOT}" -type d -path '*/attempt_*/logs/perf' -print0)

  if [[ "${found}" -eq 0 ]]; then
    echo "no logs/perf directories found under: ${ROOT}"
    rm -f "${tmp_list}"
    return 0
  fi

  run mkdir -p "$(dirname "${out}")"
  if [[ "${use_pigz}" == "1" ]]; then
    run tar -C "${ROOT}" -I "pigz -9" --null -T "${tmp_list}" -cf "${out}"
  else
    run tar -C "${ROOT}" -z --null -T "${tmp_list}" -cf "${out}"
  fi
  maybe_split_file "${out}"
  rm -f "${tmp_list}"
}

archive_xpiler_jsonl_global() {
  local out="${XPILER_JSONL_GLOBAL_OUT}"
  [[ -n "${out}" ]] || die "XPILER_JSONL_GLOBAL_OUT is empty"
  if [[ -f "${out}" ]]; then
    echo "skip (exists): ${out}"
    return 0
  fi

  local use_pigz=0
  if have pigz; then use_pigz=1; fi

  local tmp_list
  tmp_list="$(mktemp)"
  # Cleanup explicitly to avoid trap collisions.

  local found=0
  # Include any "*_xpiler.jsonl" under ROOT (typically at model/timestamp/*_xpiler.jsonl), preserving relative paths.
  while IFS= read -r -d '' f; do
    found=1
    printf '%s\0' "${f#${ROOT}/}" >>"${tmp_list}"
  done < <(find "${ROOT}" -type f -name '*_xpiler.jsonl' -print0)

  if [[ "${found}" -eq 0 ]]; then
    echo "no *_xpiler.jsonl files found under: ${ROOT}"
    rm -f "${tmp_list}"
    return 0
  fi

  run mkdir -p "$(dirname "${out}")"
  if [[ "${use_pigz}" == "1" ]]; then
    run tar -C "${ROOT}" -I "pigz -9" --null -T "${tmp_list}" -cf "${out}"
  else
    run tar -C "${ROOT}" -z --null -T "${tmp_list}" -cf "${out}"
  fi
  maybe_split_file "${out}"
  rm -f "${tmp_list}"
}

usage() {
  cat <<'EOF'
archive_xpiler.sh

用途：
  归档/汇总 xpiler 目录树下的产物，方便拷贝/传输。

关键行为：
  - 所有 tar 包都会保留相对 --root 的路径（不会写入绝对路径）。
  - --jsonl-only 会在原地生成 foo.jsonl.gz，并保留 foo.jsonl。
  - 如果系统有 pigz，会用 pigz 做压缩（更快）；否则用 gzip。
  - 生成的 tar.gz 可以按固定大小分片（见 --split-size，默认关闭）。

Options:
  --root PATH         Absolute path to xpiler root (default is repo-specific hardcoded path)
  --dry-run           Print commands without executing
  --split-size SIZE   当输出 *.tar.gz 大于 SIZE 时进行分片（默认: 0 = 不分片）。示例：5G, 500M
  --jsonl-only        逐文件压缩：把 *.jsonl -> *.jsonl.gz（产物与源文件同目录，保留 .jsonl）
  --triton-only       每个 attempt 打包：attempt_*/triton_* -> attempt_*/triton_artifacts.tar.gz
  --triton-global OUT 全局打包：收集所有 triton_* 目录 -> OUT（保留路径）
  --conv-global OUT   全局打包：收集所有 */attempt_*/logs/history/conversation.json -> OUT（保留路径）
  --retry-global OUT  全局打包：收集所有 */attempt_*/logs/retry_events.jsonl -> OUT（保留路径）
  --perf-global OUT   全局打包：收集所有 */attempt_*/logs/perf 目录 -> OUT（保留路径）
  --xpiler-jsonl-global OUT
                    全局打包：收集所有 "*_xpiler.jsonl" 文件 -> OUT（保留路径）
  --delete-after      After a tar is successfully created, delete the triton_* directories for that attempt
  -h, --help          Show this help

示例：
  # 1) 推荐先预览：
  bash scripts/archive_xpiler.sh --root /abs/path/to/xpiler --conv-global /tmp/conv.tar.gz --dry-run

  # 2) 汇总所有 triton_* 为一个大包（默认不分片）：
  bash scripts/archive_xpiler.sh --root /abs/path/to/xpiler --triton-global /tmp/triton_all.tar.gz

  # 3) 汇总所有 history/conversation.json，并开启 5G 分片：
  bash scripts/archive_xpiler.sh --root /abs/path/to/xpiler --conv-global /tmp/conv_all.tar.gz --split-size 5G

  # 4) 汇总所有 retry_events.jsonl（不分片）：
  bash scripts/archive_xpiler.sh --root /abs/path/to/xpiler --retry-global /tmp/retry_all.tar.gz

  # 5) 汇总所有 logs/perf 目录：
  bash scripts/archive_xpiler.sh --root /abs/path/to/xpiler --perf-global /tmp/perf_all.tar.gz

  # 6) 汇总所有 *_xpiler.jsonl：
  bash scripts/archive_xpiler.sh --root /abs/path/to/xpiler --xpiler-jsonl-global /tmp/xpiler_jsonl_all.tar.gz

  # 7) 每个 attempt 单独打包 triton_*：
  bash scripts/archive_xpiler.sh --root /abs/path/to/xpiler --triton-only

  # 8) 逐文件压缩（会在原地生成 *.jsonl.gz；如果你只想汇总打包，一般不需要用这个）：
  bash scripts/archive_xpiler.sh --root /abs/path/to/xpiler --jsonl-only

分片文件的合并与解压：
  cat /path/to/out.tar.gz.part* > /path/to/out.tar.gz
  tar -tzf /path/to/out.tar.gz | head
  tar -xzf /path/to/out.tar.gz -C /target/dir
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="${2:-}"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --split-size) SPLIT_SIZE="${2:-}"; shift 2;;
    --jsonl-only) DO_TRITON=0; DO_JSONL=1; shift;;
    --triton-only) DO_JSONL=0; DO_TRITON=1; shift;;
    --triton-global) TRITON_GLOBAL_OUT="${2:-}"; DO_JSONL=0; DO_TRITON=0; shift 2;;
    --conv-global) CONV_GLOBAL_OUT="${2:-}"; DO_JSONL=0; DO_TRITON=0; shift 2;;
    --retry-global) RETRY_GLOBAL_OUT="${2:-}"; DO_JSONL=0; DO_TRITON=0; shift 2;;
    --perf-global) PERF_GLOBAL_OUT="${2:-}"; DO_JSONL=0; DO_TRITON=0; shift 2;;
    --xpiler-jsonl-global) XPILER_JSONL_GLOBAL_OUT="${2:-}"; DO_JSONL=0; DO_TRITON=0; shift 2;;
    --delete-after) DELETE_AFTER=1; shift;;
    -h|--help) usage; exit 0;;
    *) die "Unknown arg: $1";;
  esac
done

[[ -n "${ROOT}" ]] || die "--root is required"
ROOT="${ROOT%/}"
[[ -d "${ROOT}" ]] || die "ROOT is not a directory: ${ROOT}"

if [[ "${DO_JSONL}" == "1" ]]; then
  compress_jsonl
fi

if [[ "${DO_TRITON}" == "1" ]]; then
  archive_triton_per_attempt
fi

if [[ -n "${TRITON_GLOBAL_OUT}" ]]; then
  archive_triton_global
fi

if [[ -n "${CONV_GLOBAL_OUT}" ]]; then
  archive_conversation_history_global
fi

if [[ -n "${RETRY_GLOBAL_OUT}" ]]; then
  archive_retry_events_global
fi

if [[ -n "${PERF_GLOBAL_OUT}" ]]; then
  archive_perf_global
fi

if [[ -n "${XPILER_JSONL_GLOBAL_OUT}" ]]; then
  archive_xpiler_jsonl_global
fi


