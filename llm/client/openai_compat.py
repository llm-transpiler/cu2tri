from enum import Enum
from logging import Logger
from openai import OpenAI, AsyncOpenAI
from utils.util import obj_to_dict as _completion_usage_to_dict
import os
import httpx
from dataclasses import dataclass
from typing import Tuple, Any, Callable
from profiler.timer import perf_counter_timestamp_ns, ns_to_ms, TimerSample


class TTFTTracker:
    """Elegant TTFT (Time To First Token) tracker using direct timestamp measurement."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.first_token_received = False
        self.request_start_ns = None
        self.first_token_ns = None

    def start_request(self) -> None:
        """Start timing the LLM request."""
        if self.enabled:
            self.request_start_ns = perf_counter_timestamp_ns()
            self.first_token_received = False
            self.first_token_ns = None

    def mark_first_token(self) -> None:
        """Mark when the first token is received (for TTFT calculation)."""
        if self.enabled and not self.first_token_received and self.request_start_ns:
            self.first_token_ns = perf_counter_timestamp_ns()
            self.first_token_received = True

    def end_request(self) -> Tuple[float, float | None]:
        """End the request timing and return (total_time, ttft)."""
        if not self.enabled or not self.request_start_ns:
            return None, None

        end_ns = perf_counter_timestamp_ns()
        total_time = ns_to_ms(end_ns - self.request_start_ns)

        ttft = None
        if self.first_token_ns:
            ttft = ns_to_ms(self.first_token_ns - self.request_start_ns)

        return ttft, total_time

    def get_time_track(self) -> tuple[float | None, float | None]:
        ttft_ms, total_time_ms = self.end_request()
        return ttft_ms, total_time_ms


async def _fetch_openrouter_generation_info(gen_id: str, logger: Logger = None):
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY or not gen_id:
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://openrouter.ai/api/v1/generation",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                params={"id": gen_id},
            )
            resp.raise_for_status()
            return resp.json().get("data")
    except Exception:
        if logger:
            logger.debug("Failed to fetch OpenRouter generation info", exc_info=True)
        return None


async def async_openai_llm_call(async_client: AsyncOpenAI, api_params: dict[str, Any] = {}, logger: Logger = None,
                                ttft_tracker: TTFTTracker | None = None):
    if ttft_tracker is None:
        ttft_tracker = TTFTTracker(enabled=True)

    ttft_tracker.start_request()

    is_stream = api_params.get('stream', False)
    completion = await async_client.chat.completions.create(**api_params)
    resp_content = ""
    resp_reasoning_content = ""
    model_used = None
    usage_dict = None
    gen_id = None

    # Check if this is an OpenRouter serviceLLMTimeTracker
    is_openrouter = False
    if hasattr(async_client, 'base_url') and async_client.base_url:
        is_openrouter = 'openrouter.ai' in str(async_client.base_url)

    if is_stream:
        # Handle streaming response
        first_content_received = False
        async for chunk in completion:
            # Get model from first chunk
            if model_used is None and hasattr(chunk, 'model'):
                model_used = chunk.model
            if gen_id is None and hasattr(chunk, 'id'):
                gen_id = chunk.id
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content is not None:
                    content = delta.content
                    # Mark first token for TTFT
                    if not first_content_received and content.strip():
                        ttft_tracker.mark_first_token()
                        first_content_received = True
                    resp_content += content

                elif hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content = delta.reasoning_content
                    # Mark first token for TTFT
                    if not first_content_received and reasoning_content.strip():
                        ttft_tracker.mark_first_token()
                        first_content_received = True
                    resp_reasoning_content += reasoning_content

            if hasattr(chunk, 'usage') and chunk.usage is not None:
                try:
                    usage_dict = _completion_usage_to_dict(chunk.usage)
                except Exception:
                    if logger:
                        logger.debug("Failed to convert streaming usage to dict", exc_info=True)
    else:
        # For non-streaming, TTFT is when we get the complete response
        ttft_tracker.mark_first_token()
        # Handle non-streaming response
        if not completion.choices or len(completion.choices) == 0:
            raise ValueError("API returned empty choices list")
        if hasattr(completion.choices[0].message, 'reasoning_content'):
           resp_reasoning_content = completion.choices[0].message.reasoning_content
        else:
            resp_reasoning_content = None
        if hasattr(completion.choices[0].message, 'content'):
            resp_content = completion.choices[0].message.content
        else:
            resp_content = None
        model_used = completion.model if hasattr(completion, 'model') else None
        gen_id = getattr(completion, 'id', None)

        if hasattr(completion, 'usage') and completion.usage is not None:
            try:
                usage_dict = _completion_usage_to_dict(completion.usage)
            except Exception:
                if logger:
                    logger.debug("Failed to convert usage to dict", exc_info=True)

    if resp_reasoning_content is not None:
        resp_reasoning_content = resp_reasoning_content.strip()
    if resp_content is not None:
        resp_content = resp_content.strip()

    extra_info = None
    if gen_id and is_openrouter:
        extra_info = await _fetch_openrouter_generation_info(gen_id, logger)

    ttft_ms, total_time_ms = ttft_tracker.get_time_track()
    if usage_dict is None:
        usage_dict = {}
    if ttft_ms is not None:
        usage_dict['tracked_ttft_ms'] = ttft_ms
    if total_time_ms is not None:
        usage_dict['tracked_total_time_ms'] = total_time_ms

    return resp_reasoning_content, resp_content, model_used, usage_dict, extra_info


class CallingIdentifier(Enum):
    DEEPSEEK_OPENAI = 'deepseek_openai'
    CLAUDE_OPENROUTER = 'claude_openrouter'
    OPENAI_OFFICIAL = 'openai_official'
    GEMINI_OPENAI = 'gemini_openai'
    ANTHROPIC_OFFICIAL = 'anthropic_official'
    ANTHROPIC_OPENROUTER = 'anthropic_openrouter'
    OPENAI_OPENROUTER = 'openai_openrouter'


def get_api_param_gemini_openai(
    messages: list,
    model_name: str = "gemini-2.5-pro",
    stream: bool = True,
    temperature: float = 0.35,
    max_tokens: int = 65536,
    thinking: bool = True,
    thinking_budget: int = -1,
    include_thoughts: bool = False,
):
    if stream:
        api_param = {
            'model': model_name,
            'messages': messages,
            'stream': stream,
            'stream_options': {"include_usage": True},
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
    else:
        api_param = {
            'model': model_name,
            'messages': messages,
            'stream': stream,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
    if thinking:
        api_param['extra_body'] = {
            'extra_body': {
                "google": {
                    "thinking_config": {
                        "thinking_budget": thinking_budget,
                        "include_thoughts": include_thoughts,
                    }
                }
            }
        }
    return api_param


def get_api_param_openai_default(
    messages: list,
    model_name: str = "gpt-4o",
    stream: bool = True,
    temperature: float = 0.35,
    max_tokens: int = 65536,
):
    if stream:
        return {
            'model': model_name,
            'messages': messages,
            'stream': stream,
            'stream_options': {"include_usage": True},
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
    else:
        return {
            'model': model_name,
            'messages': messages,
            'stream': stream,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }


def get_api_param_deepseek_openai(
    messages: list,
    model_name: str = "deepseek-reasoner",
    stream: bool = True,
    temperature: float = 0.35,
    max_tokens: int = 65536,
):
    # DeepSeek Reasoner comes with built-in thinking; no extra params required per docs
    return {
        'model': model_name,
        'messages': messages,
        'stream': stream,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }


def get_api_param_anthropic_openrouter(
    messages: list,
    model_name: str = "anthropic/claude-4-sonnet",
    stream: bool = True,
    temperature: float = 0.35,
    max_tokens: int = 65536,
    reasoning_effort: str = "high",
):
    if stream:
        return {
            'model': model_name,
            'messages': messages,
            'stream': stream,
            'stream_options': {"include_usage": True},
            'temperature': temperature,
            'max_tokens': max_tokens,
            'reasoning_effort': reasoning_effort,
        }
    else:
        return {
            'model': model_name,
            'messages': messages,
            'stream': stream,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'reasoning_effort': reasoning_effort,
        }


def get_api_param_openai_openrouter(
    messages: list,
    model_name: str = "openai/gpt-5",
    stream: bool = True,
    temperature: float = 0.35,
    max_tokens: int = 65536,
    reasoning_effort: str = "high",
):
    if stream:
        return {
            'model': model_name,
            'messages': messages,
            'stream': stream,
            'stream_options': {"include_usage": True},
            'temperature': temperature,
            'max_tokens': max_tokens,
            'reasoning_effort': reasoning_effort,
        }
    else:
        return {
            'model': model_name,
            'messages': messages,
            'stream': stream,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'reasoning_effort': reasoning_effort,
        }


API_PARAMS_DICT = {
    CallingIdentifier.GEMINI_OPENAI: get_api_param_gemini_openai,
    CallingIdentifier.DEEPSEEK_OPENAI: get_api_param_deepseek_openai,
    # CallingIdentifier.ANTHROPIC_OFFICIAL: get_api_param_anthropic_official,
    CallingIdentifier.ANTHROPIC_OPENROUTER: get_api_param_anthropic_openrouter,
    CallingIdentifier.OPENAI_OFFICIAL: get_api_param_openai_default,
    CallingIdentifier.OPENAI_OPENROUTER: get_api_param_openai_openrouter,
}


def get_api_params_method(callId: CallingIdentifier) -> Callable:
    return API_PARAMS_DICT.get(callId, lambda *args, **kwargs: {})


def make_openai_single_message(role: str, content: str):
    return {"role": role, "content": content}


def make_openai_message_system(content: str):
    return make_openai_single_message("system", content)


def make_openai_message_user(content: str):
    return make_openai_single_message("user", content)


def make_openai_message_assistant(content: str):
    return make_openai_single_message("assistant", content)


def create_ttft_tracker(enabled: bool = True) -> TTFTTracker:
    return TTFTTracker(enabled=enabled)


__all__ = [
    # core call helpers
    'async_openai_llm_call',
    # timing and profiling
    'TTFTTracker', 'create_ttft_tracker',
    # enums and routing
    'CallingIdentifier', 'get_api_params_method',
    # api param builders
    'get_api_param_openai_default', 'get_api_param_openai_openrouter',
    'get_api_param_gemini_openai', 'get_api_param_deepseek_openai',
    'get_api_param_anthropic_openrouter',
    # message helpers
    'make_openai_single_message', 'make_openai_message_system',
    'make_openai_message_user', 'make_openai_message_assistant',
]