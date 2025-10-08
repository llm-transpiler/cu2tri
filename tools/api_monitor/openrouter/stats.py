import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import random
from tools.api_monitor.openrouter import query as or_query
from pydantic import BaseModel, Field
import os
import dotenv
import logging
dotenv.load_dotenv()
DEFAULT_LOGGER = logging.getLogger(__name__)


class Meta(BaseModel):
    model: str = Field(description="model exact name")
    gen_id: str = Field(description="OpenRouter generation id")
    finish_reason: str | None = Field(default=None, description="finish reason")


class Timings(BaseModel):
    latency_ms: int | None = None
    generation_time_ms: int | None = None
    manual_ttft_ms: int | None = None
    manual_req_e2e_ms: int | None = None


class TokensNative(BaseModel):
    prompt: int | None = None
    completion: int | None = None
    reasoning: int | None = None


class TokensStats(BaseModel):
    native_tokens_prompt: int | None = None
    native_tokens_completion: int | None = None
    native_tokens_reasoning: int | None = None


class EstimateCost(BaseModel):
    scale: str | None = '1m'
    cost_prompt: float | None = 0.0
    cost_completion: float | None = 0.0
    cost_cache_read: float | None = 0.0
    cost_cache_write: float | None = 0.0
    total_estimated_cost: float | None = 0.0


class Pricing(BaseModel):
    input: float | None = None
    output: float | None = None
    cache_read: float | None = None
    cache_write: float | None = None


class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: None | dict = None
    prompt_tokens_details: None | dict = None


class Report(BaseModel):
    meta: Meta
    timings: Timings
    tokens: TokensStats
    completion_usage: dict | None = None
    token_per_sec: float | None = None
    usage: float | None = None
    total_cost: float | None = None


def create_session_with_retries(
    total_retries: int = 3, backoff_factor: float = 0.5
) -> requests.Session:
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods={"GET", "POST"},
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def report_stats(
    gen_id: str,
    model_used: str,
    manual_req_e2e_ms: int | None = None,
    manual_ttft_ms: int | None = None,
    completion_usage: dict | None = None,
    finish_reason_collected: str | None = None,
    logger: logging.Logger | None = DEFAULT_LOGGER,
) -> Report:
    session = create_session_with_retries()

    def _fetch_once() -> dict:
        try:
            response = session.get(
                "https://openrouter.ai/api/v1/generation",
                headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
                params={"id": gen_id},
                timeout=(5, 20),
            )
            return response.json() if response.ok else {}
        except Exception:
            return {}

    def _is_ready(data: dict) -> bool:
        if not data:
            return False
        d = data.get('data') or {}
        # Consider ready if generation_time or token counts are present
        if d.get('generation_time') is not None:
            return True
        if d.get('native_tokens_completion') is not None or d.get('native_tokens_prompt') is not None:
            return True
        if d.get('usage') is not None or d.get('total_cost') is not None:
            return True
        return False

    # short polling for the same gen_id to wait until stats are populated
    max_attempts = int(os.getenv('OPENROUTER_GEN_POLL_ATTEMPTS', '5'))
    min_sleep_ms = int(os.getenv('OPENROUTER_GEN_POLL_MIN_MS', '800'))
    max_sleep_ms = int(os.getenv('OPENROUTER_GEN_POLL_MAX_MS', '1500'))

    stats = {}
    for _ in range(max_attempts):
        stats = _fetch_once()
        if _is_ready(stats):
            break
        # jittered sleep between 0.8s and 1.5s by default
        sleep_ms = random.randint(min_sleep_ms, max_sleep_ms)
        time.sleep(sleep_ms / 1000.0)

    data = stats.get('data', {}) or {}
    if not data:
        logger.error("data get failed")

    latency = data.get('latency')
    generation_time = data.get('generation_time')

    native_tokens_prompt = data.get('native_tokens_prompt')
    native_tokens_completion = data.get('native_tokens_completion')
    native_tokens_reasoning = data.get('native_tokens_reasoning')

    native_tokens_per_sec = None
    try:
        native_tokens_per_sec = float(native_tokens_completion) / (float(generation_time) / 1000.0)
    except Exception:
        if logger:
            logger.error("native_tokens_per_sec calculation failed")

    # Prefer in-process collected finish_reason; log if mismatch with API value
    api_finish_reason = data.get('finish_reason')
    final_finish_reason = finish_reason_collected if finish_reason_collected is not None else api_finish_reason
    if finish_reason_collected is not None and api_finish_reason is not None and finish_reason_collected != api_finish_reason:
        if logger:
            logger.warning(
                "finish_reason mismatch: collected=%s, api=%s",
                finish_reason_collected,
                api_finish_reason,
            )

    report = Report(
        meta=Meta(model=model_used, gen_id=gen_id, finish_reason=final_finish_reason),
        timings=Timings(
            latency_ms=latency,
            generation_time_ms=generation_time,
            manual_ttft_ms=manual_ttft_ms,
            manual_req_e2e_ms=manual_req_e2e_ms,
        ),
        tokens=TokensStats(
            native_tokens_prompt=native_tokens_prompt,
            native_tokens_completion=native_tokens_completion,
            native_tokens_reasoning=native_tokens_reasoning,
        ),
        completion_usage=(completion_usage if isinstance(completion_usage, dict) else None),
        token_per_sec=native_tokens_per_sec,
        total_cost=data.get('total_cost'),
        usage=data.get('usage'),
    )
    return report


def estimate_cost(model_name: str, input_tokens_count: int, output_tokens_count: int) -> EstimateCost:
    try:
        row = or_query.query_by_exact_name(model_name)
        if row:
            pricing = Pricing(
                input=row.get('input_price'),
                output=row.get('output_price'),
                cache_read=row.get('cache_read_price'),
                cache_write=row.get('cache_write_price'),
            )
            est = or_query.estimate_cost(
                pricing.model_dump(),
                prompt_tokens=input_tokens_count,
                completion_tokens=output_tokens_count,
                price_scale='per_1m',
            )
            return EstimateCost(**(est or {}))
    except Exception:
        pass
    return EstimateCost()
