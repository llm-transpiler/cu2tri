import argparse
import csv
import json
import os
from typing import Dict, Any, List, Optional, Tuple
import requests


DEFAULT_CSV_PATH = "/workspace/tools/price/openrouter_api.csv"
DEFAULT_CONFIG_PATH = "/workspace/tools/api_monitor/openrouter/models_config.json"


CSV_HEADER_KEYS = {
    "Input Price (OpenRouter)": "input_price",
    "Output Price (OpenRouter)": "output_price",
    "Cache Read Price (OpenRouter)": "cache_read_price",
    "Cache Write Price (OpenRouter)": "cache_write_price",
    "Max Output": "max_output",
    "Total Context": "total_context",
    "endpoint version": "endpoint_version",
    "Latency": "latency",
    "Output/Input Price": "output_input_ratio",
    "Openrouter Card": "openrouter_card",
    "API Provider": "api_provider",
    "Exact Name": "exact_name",
    "Access": "access",
    "Model Vendor": "model_vendor",
    "Throughput": "throughput",
    "token section": "token_section",
    "Model Name": "model_name",
}


NUMERIC_FLOAT_FIELDS = {
    "input_price",
    "output_price",
    "cache_read_price",
    "cache_write_price",
    "latency",
    "output_input_ratio",
    "throughput",
}


NUMERIC_INT_FIELDS = {
    "max_output",
    "total_context",
}


def _to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(value: str) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def load_openrouter_csv(csv_path: str = DEFAULT_CSV_PATH) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row: Dict[str, Any] = {}
            # Map headers to canonical keys
            for header, key in CSV_HEADER_KEYS.items():
                if header in raw_row:
                    row[key] = raw_row.get(header)
            # Type conversions
            for key in list(row.keys()):
                if key in NUMERIC_FLOAT_FIELDS:
                    row[key] = _to_float(row[key])
                elif key in NUMERIC_INT_FIELDS:
                    row[key] = _to_int(row[key])
                else:
                    # Normalize strings
                    if isinstance(row[key], str):
                        row[key] = row[key].strip()
            rows.append(row)
    return rows


def _price_scale_to_tokens(scale: str) -> int:
    s = (scale or "per_1m").strip().lower()
    if s in {"per_token", "token", "per_1"}:
        return 1
    if s in {"per_1k", "per_1000", "1k"}:
        return 1000
    # default to per 1,000,000 tokens, which matches typical OpenRouter pricing
    return 1_000_000


def estimate_cost(
    pricing: Dict[str, Any],
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    cache_read_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
    price_scale: str = "per_1m",
) -> Optional[Dict[str, Any]]:
    if not pricing:
        return None
    scale_tokens = _price_scale_to_tokens(price_scale)

    input_price = pricing.get("input")
    output_price = pricing.get("output")
    cache_read_price = pricing.get("cache_read")
    cache_write_price = pricing.get("cache_write")

    def part_cost(tokens: Optional[int], unit_price: Optional[float]) -> Optional[float]:
        if tokens is None or unit_price is None:
            return None
        try:
            return unit_price * (float(tokens) / float(scale_tokens))
        except Exception:
            return None

    cost_prompt = part_cost(prompt_tokens, input_price)
    cost_completion = part_cost(completion_tokens, output_price)
    cost_cache_read = part_cost(cache_read_tokens, cache_read_price)
    cost_cache_write = part_cost(cache_write_tokens, cache_write_price)

    total_cost = 0.0
    for v in [cost_prompt, cost_completion, cost_cache_read, cost_cache_write]:
        if isinstance(v, (int, float)):
            total_cost += float(v)

    return {
        "scale": price_scale,
        "cost_prompt": cost_prompt,
        "cost_completion": cost_completion,
        "cost_cache_read": cost_cache_read,
        "cost_cache_write": cost_cache_write,
        "total_estimated_cost": total_cost,
    }


def compute_max_allowed_completion_tokens(
    limits: Dict[str, Any], prompt_tokens: Optional[int]
) -> Optional[int]:
    if not limits:
        return None
    max_output = limits.get("max_output")
    total_context = limits.get("total_context")

    # If total context is known, compute remaining capacity after prompt
    remaining_from_context: Optional[int] = None
    if isinstance(total_context, int) and isinstance(prompt_tokens, int):
        remaining_from_context = max(0, total_context - prompt_tokens)

    # If max_output is known, it's an upper bound
    if isinstance(max_output, int) and isinstance(remaining_from_context, int):
        return max(0, min(max_output, remaining_from_context))
    if isinstance(max_output, int):
        return max(0, max_output)
    if isinstance(remaining_from_context, int):
        return max(0, remaining_from_context)
    return None


def fetch_generation_stats(gen_id: str, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not gen_id:
        return None
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/generation",
            headers={"Authorization": f"Bearer {key}"},
            params={"id": gen_id},
            timeout=20,
        )
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def compare_estimate_with_stats(
    estimate: Dict[str, Any], stats: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not estimate or not stats:
        return None
    data = stats.get("data", {}) or {}
    observed_total_cost = data.get("total_cost")
    observed_tokens_prompt = data.get("tokens_prompt")
    observed_tokens_completion = data.get("tokens_completion")

    result: Dict[str, Any] = {
        "observed_total_cost": observed_total_cost,
        "observed_tokens_prompt": observed_tokens_prompt,
        "observed_tokens_completion": observed_tokens_completion,
    }

    try:
        est_total = float(estimate.get("total_estimated_cost"))
        obs_total = float(observed_total_cost)
        result["absolute_delta_cost"] = abs(est_total - obs_total)
        result["relative_delta_cost"] = (
            abs(est_total - obs_total) / obs_total if obs_total != 0 else None
        )
    except Exception:
        pass

    return result


def normalize_provider_name(api_provider: str) -> str:
    if not api_provider:
        return "unknown"
    provider = api_provider.strip().lower()
    mapping = {
        "openai": "openai",
        "azure": "azure",
        "google ai studio": "google_ai_studio",
        "google vertex": "google_vertex",
        "google vertex (global)": "google_vertex",
        "anthropic": "anthropic",
        "deepinfra": "deepinfra",
        "siliconflow": "siliconflow",
        "xai": "xai",
        "x-ai": "xai",
        "z.ai": "zai",
        "z.ai (zhipu)": "zai",
        "z.ai (zhipu ai)": "zai",
        "z.ai ltd": "zai",
        "z.ai limited": "zai",
        "alibaba cloud int.": "alibaba_cloud",
        "alibaba cloud": "alibaba_cloud",
    }
    return mapping.get(provider, provider.replace(" ", "_"))


def preferred_provider_order_for_vendor(model_vendor: str) -> List[str]:
    vendor = (model_vendor or "").strip().lower()
    preferences: Dict[str, List[str]] = {
        "openai": ["openai"],
        "anthropic": ["anthropic"],
        "google": ["google_ai_studio"],
        "qwen": ["alibaba_cloud"],
        "deepseek": ["deepinfra"],
        "x-ai": ["xai"],
        "xai": ["xai"],
        "z-ai": ["zai"],
        "zai": ["zai"],
    }
    return preferences.get(vendor, [])


def _row_sort_key_for_best(r: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Sorting key for selecting the best row with tie-breakers:
    1) input_price asc (None last)
    2) latency asc (None last)
    3) output_price asc (None last)
    4) throughput desc (None last)
    """
    input_price = r.get("input_price")
    latency = r.get("latency")
    output_price = r.get("output_price")
    throughput = r.get("throughput")

    return (
        input_price is None, input_price if input_price is not None else 1e18,
        latency is None, latency if latency is not None else 1e18,
        output_price is None, output_price if output_price is not None else 1e18,
        throughput is None, -(throughput if throughput is not None else -1e18),
    )


def select_best_row(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    # Try vendor-preferred providers first
    vendor = rows[0].get("model_vendor", "")
    preferences = preferred_provider_order_for_vendor(vendor)
    if preferences:
        rows_with_norm = [
            (normalize_provider_name(r.get("api_provider", "")), r) for r in rows
        ]
        for pref in preferences:
            pref_rows = [r for prov, r in rows_with_norm if prov == pref]
            if pref_rows:
                # If multiple, choose with tie-breakers
                pref_rows.sort(key=_row_sort_key_for_best)
                return pref_rows[0]

    # Fallback: choose the row with minimum input price
    rows_sorted = sorted(rows, key=_row_sort_key_for_best)
    return rows_sorted[0]


def query_by_exact_name(exact_name: str, csv_path: str = DEFAULT_CSV_PATH, all_matches: bool = False) -> Any:
    data = load_openrouter_csv(csv_path)
    matches = [r for r in data if (r.get("exact_name") or "").lower() == exact_name.strip().lower()]
    if not matches:
        return None
    if all_matches:
        # Sort by input price ascending for display
        matches.sort(key=lambda r: (r.get("input_price") is None, r.get("input_price", 1e18)))
        return matches
    return select_best_row(matches)


def build_models_config(
    csv_path: str = DEFAULT_CSV_PATH,
    provider_whitelist: Optional[List[str]] = None,
    include_all_providers: bool = False,
) -> Dict[str, Any]:
    """
    Build a consolidated models configuration from the OpenRouter CSV.

    Parameters:
    - csv_path: path to the OpenRouter pricing CSV
    - provider_whitelist: optional list of normalized provider names to include
      (e.g., ["openai", "deepinfra"]). If None, all providers are considered.
    - include_all_providers: if True, include all provider variants per model in
      the output under the "providers" field. If False, only the selected best
      entry (backward compatible behavior).
    """
    data = load_openrouter_csv(csv_path)

    # Normalize whitelist for comparisons
    whitelist_norm: Optional[set] = None
    if provider_whitelist:
        whitelist_norm = {normalize_provider_name(p) for p in provider_whitelist}

    # Group by exact name
    by_exact: Dict[str, List[Dict[str, Any]]] = {}
    for row in data:
        exact = row.get("exact_name")
        if not exact:
            continue
        prov_norm = normalize_provider_name(row.get("api_provider", ""))
        if whitelist_norm is not None and prov_norm not in whitelist_norm:
            continue
        by_exact.setdefault(exact, []).append(row)

    config: Dict[str, Any] = {
        "supported_models": [],
        "models": {}
    }

    for exact, rows in by_exact.items():
        if not rows:
            continue

        # Determine best row using existing vendor preference + min input price
        best = select_best_row(rows)
        if not best:
            continue

        best_provider_key = normalize_provider_name(best.get("api_provider", ""))

        # Backward-compatible top-level (default selection)
        model_entry: Dict[str, Any] = {
            "exact_name": exact,
            "name": best.get("model_name"),
            "pricing": {
                "input": best.get("input_price"),
                "output": best.get("output_price"),
                "cache_read": best.get("cache_read_price"),
                "cache_write": best.get("cache_write_price"),
            },
            "limits": {
                "max_output": best.get("max_output"),
                "total_context": best.get("total_context"),
            },
            "model_vendor": best.get("model_vendor"),
            "provider": best_provider_key,
            "endpoint_version": best.get("endpoint_version"),
            "openrouter_card": best.get("openrouter_card"),
            "access": best.get("access"),
        }

        if include_all_providers:
            # Build providers aggregation (one best entry per provider)
            providers_map: Dict[str, Dict[str, Any]] = {}

            # Group by normalized provider
            by_provider: Dict[str, List[Dict[str, Any]]] = {}
            for r in rows:
                pkey = normalize_provider_name(r.get("api_provider", ""))
                by_provider.setdefault(pkey, []).append(r)

            for pkey, prows in by_provider.items():
                # Select best row per provider using tie-breakers
                prows_sorted = sorted(prows, key=_row_sort_key_for_best)
                best_row = prows_sorted[0]

                providers_map[pkey] = {
                    "pricing": {
                        "input": best_row.get("input_price"),
                        "output": best_row.get("output_price"),
                        "cache_read": best_row.get("cache_read_price"),
                        "cache_write": best_row.get("cache_write_price"),
                    },
                    "limits": {
                        "max_output": best_row.get("max_output"),
                        "total_context": best_row.get("total_context"),
                    },
                    "endpoint_version": best_row.get("endpoint_version"),
                    "openrouter_card": best_row.get("openrouter_card"),
                    "access": best_row.get("access"),
                    "api_provider_raw": best_row.get("api_provider"),
                }

            model_entry["default_provider"] = best_provider_key
            model_entry["providers"] = providers_map

        config["models"][exact] = model_entry
        config["supported_models"].append(exact)

    config["supported_models"].sort()
    return config


def save_models_config(config: Dict[str, Any], out_path: str = DEFAULT_CONFIG_PATH) -> str:
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    '''
    # python /workspace/tools/api_monitor/openrouter/query.py --exact-name openai/gpt-5
    # python /workspace/tools/api_monitor/openrouter/query.py --exact-name anthropic/claude-sonnet-4 --all
    # python /workspace/tools/api_monitor/openrouter/query.py --generate-config --csv /workspace/tools/price/openrouter_api.csv --out /workspace/tools/api_monitor/openrouter/models_config.json
    仅生成最佳provider（默认行为，兼容旧结构）:
    python /workspace/tools/api_monitor/openrouter/query.py --generate-config --csv /workspace/tools/price/openrouter_api.csv --out /workspace/tools/api_monitor/openrouter/models_config.json
    只包含指定provider（如 openai, deepinfra），仍仅选最佳项:
    python /workspace/tools/api_monitor/openrouter/query.py --generate-config --csv /workspace/tools/price/openrouter_api.csv --out /workspace/tools/api_monitor/openrouter/models_config.json --providers openai,deepinfra
    聚合所有provider到同一模型下（含最佳与所有可选项）:
    python /workspace/tools/api_monitor/openrouter/query.py --generate-config --csv /workspace/tools/price/openrouter_api.csv --out /workspace/tools/api_monitor/openrouter/models_config.json --include-all-providers
    同时过滤provider并聚合:
    python /workspace/tools/api_monitor/openrouter/query.py --generate-config --csv /workspace/tools/price/openrouter_api.csv --out /workspace/tools/api_monitor/openrouter/models_config.json --providers openai,deepinfra --include-all-providers
    '''
    parser = argparse.ArgumentParser(description="OpenRouter pricing query tool")
    parser.add_argument("--csv", dest="csv_path", default=DEFAULT_CSV_PATH, help="Path to openrouter_api.csv")
    parser.add_argument("--exact-name", dest="exact_name", default=None, help="Exact model name to query (e.g., 'openai/gpt-5')")
    parser.add_argument("--all", dest="all_matches", action="store_true", help="Show all provider matches for the exact name")
    parser.add_argument("--generate-config", dest="generate_config", action="store_true", help="Generate models_config.json from CSV")
    parser.add_argument("--out", dest="out_path", default=DEFAULT_CONFIG_PATH, help="Output path for generated config JSON")
    parser.add_argument(
        "--providers",
        dest="providers",
        default=None,
        help="Comma-separated list of providers to include when generating config (normalized, e.g. 'openai,deepinfra')",
    )
    parser.add_argument(
        "--include-all-providers",
        dest="include_all_providers",
        action="store_true",
        help="Include all provider variants per model in generated config",
    )

    # Estimation and verification args
    parser.add_argument("--prompt-tokens", dest="prompt_tokens", type=int, default=None, help="Prompt tokens for cost estimation")
    parser.add_argument("--completion-tokens", dest="completion_tokens", type=int, default=None, help="Completion tokens for cost estimation")
    parser.add_argument("--cache-read-tokens", dest="cache_read_tokens", type=int, default=None, help="Cache read tokens for cost estimation")
    parser.add_argument("--cache-write-tokens", dest="cache_write_tokens", type=int, default=None, help="Cache write tokens for cost estimation")
    parser.add_argument("--price-scale", dest="price_scale", default="per_1m", choices=["per_token", "per_1k", "per_1m"], help="Unit of prices in CSV")
    parser.add_argument("--gen-id", dest="gen_id", default=None, help="OpenRouter generation id to verify against")
    parser.add_argument("--api-key", dest="api_key", default=None, help="OpenRouter API key (overrides env)")

    args = parser.parse_args()

    if args.generate_config:
        provider_whitelist = None
        if args.providers:
            provider_whitelist = [p.strip() for p in args.providers.split(",") if p.strip()]
        config = build_models_config(
            args.csv_path,
            provider_whitelist=provider_whitelist,
            include_all_providers=args.include_all_providers,
        )
        out_file = save_models_config(config, args.out_path)
        print(json.dumps({"ok": True, "config_path": out_file, "num_models": len(config.get("supported_models", []))}, indent=2))
        return

    if not args.exact_name:
        raise SystemExit("--exact-name is required unless --generate-config is used")

    result = query_by_exact_name(args.exact_name, args.csv_path, all_matches=args.all_matches)
    if result is None:
        print(json.dumps({"ok": False, "error": "model_not_found", "exact_name": args.exact_name}, indent=2))
        return

    if isinstance(result, list):
        # All matches
        for r in result:
            r = dict(r)
            r["provider"] = normalize_provider_name(r.get("api_provider", ""))
        print(json.dumps({"ok": True, "matches": result}, indent=2))
    else:
        r = dict(result)
        r["provider"] = normalize_provider_name(r.get("api_provider", ""))
        # Keep only key fields for concise output
        concise = {
            "exact_name": r.get("exact_name"),
            "provider": r.get("provider"),
            "model_vendor": r.get("model_vendor"),
            "pricing": {
                "input": r.get("input_price"),
                "output": r.get("output_price"),
                "cache_read": r.get("cache_read_price"),
                "cache_write": r.get("cache_write_price"),
            },
            "limits": {
                "max_output": r.get("max_output"),
                "total_context": r.get("total_context"),
            },
            "access": r.get("access"),
        }

        # Compute estimate if token counts are provided
        estimate = None
        if args.prompt_tokens is not None or args.completion_tokens is not None or args.cache_read_tokens is not None or args.cache_write_tokens is not None:
            estimate = estimate_cost(
                concise.get("pricing", {}),
                prompt_tokens=args.prompt_tokens,
                completion_tokens=args.completion_tokens,
                cache_read_tokens=args.cache_read_tokens,
                cache_write_tokens=args.cache_write_tokens,
                price_scale=args.price_scale,
            )

        # Compute max allowed completion tokens given prompt tokens
        max_allowed = compute_max_allowed_completion_tokens(concise.get("limits", {}), args.prompt_tokens)

        # Optional verification against OpenRouter generation stats
        verification = None
        stats = None
        if args.gen_id:
            stats = fetch_generation_stats(args.gen_id, api_key=args.api_key)
            if estimate and stats:
                verification = compare_estimate_with_stats(estimate, stats)

        payload = {"ok": True, "result": concise}
        if estimate is not None:
            payload["estimate"] = estimate
        if max_allowed is not None:
            payload["max_allowed_completion_tokens"] = max_allowed
        if stats is not None:
            payload["openrouter_generation_stats"] = stats
        if verification is not None:
            payload["verification"] = verification

        print(json.dumps(payload, indent=2))
        print(json.dumps({"ok": True, "result": concise}, indent=2))


if __name__ == "__main__":
    main()


