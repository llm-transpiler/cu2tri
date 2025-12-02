# NVGPU Performance Runner

`run_perf_tests.py` submits performance runs for `check_triton_gpu_all.py`
artifacts through the NVGPU service. Every attempt copies its NVGPU task log
into `.../attempt_xx/logs/perf/`, while the runner itself mirrors a session log
inside `cu2til/llm_trans/perf/logs/`.

## Basic usage

```bash
python -m cu2til.llm_trans.perf.run_perf_tests \
  --project cu2tri \
  --pipeline xpiler \
  --model-tag deepseek_v3_2_exp \
  --run-timestamp 20251024_075645 \
  --nvgpu-server http://localhost:8080
```

Key defaults:

- `task_type=performance`
- `task_mode=shared`
- `max_concurrency=10` (number of NVGPU tasks in flight)
- Runner logs: `cu2til/llm_trans/perf/logs/perf_<timestamp>.log` plus `perf.log`
  (override with `--log-dir /tmp/my_perf_logs`).
- Attempt logs: each `attempt_xx/logs/perf/` contains the NVGPU task log copied
  verbatim (`perf_<timestamp>.log` + `perf.log`). Use `--attempt-log-dir DIR`
  if you want all NVGPU logs mirrored into a single directory.

## Filtering

Use any combination of the following flags to narrow the run list:

- `--case-types add conv2d`
- `--case-names add_1_15_64`
- `--case-name-patterns add_*_64`
- `--case-name-substrings add_1 15_64`
- `--attempts 1 2 3`
- `--min-attempt 2 --max-attempt 4`
- `--limit 10`

`--dry-run` prints the matched attempts without launching NVGPU jobs.

## Multi-run sweep

- `--all-model-tags` iterates every model tag under the selected project/pipeline.
- `--all-run-timestamps` iterates every timestamp shared by the runs/stats trees
  for each model tag. Without this flag the newest timestamp is chosen.
- When multiple model tags exist and neither a specific `--model-tag` nor
  `--all-model-tags` is provided, the script aborts to avoid ambiguity.
- `--stats-file` targets a single run and therefore cannot be combined with the
  `--all-*` flags.

## Concurrency

All matched attempts are submitted concurrently (bounded by
`--max-concurrency`) so multiple Triton kernels can run in parallel on the
NVGPU server. Increase or decrease the limit to match the available GPU
capacity. A health check runs before submissions to ensure the server is up.

## Logs

- Runner/system log: planning, submission and summary info (same as console
  output) under `--log-dir` (defaults to `perf/logs/`).
- Attempt log: NVGPU task log copied from `server/nvgpu/logs/tasks/*.log` into
  the attempt's `logs/perf/` directory (or `--attempt-log-dir` if provided).

## Extra script arguments

Forward custom arguments to each `check_triton_gpu_all.py` invocation with
`--script-arg`, e.g. `--script-arg --compile-only`.
