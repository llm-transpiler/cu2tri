from enum import Enum
from logging import Logger
from openai import OpenAI, AsyncOpenAI
from utils.util import obj_to_dict as _completion_usage_to_dict
import os
import httpx

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def openai_llm_call(client: OpenAI, api_params={}, logger: Logger = None):
    """Synchronous OpenAI-compatible chat call. Returns content string."""
    is_stream = api_params.get('stream', False)
    completion = client.chat.completions.create(**api_params)
    resp_content = ""
    if is_stream:
        for chunk in completion:
            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                resp_content += content
    else:
        if not completion.choices or len(completion.choices) == 0:
            raise ValueError("API returned empty choices list")
        resp_content = completion.choices[0].message.content
    return resp_content.strip()


async def _fetch_openrouter_generation_info(gen_id: str, logger: Logger = None):
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


async def async_openai_llm_call(async_client: AsyncOpenAI, api_params={}, logger: Logger = None):
    """Async version of openai_llm_call. Returns tuple of (content, model_used, usage_dict, openrouter_info)."""
    is_stream = api_params.get('stream', False)
    completion = await async_client.chat.completions.create(**api_params)
    resp_content = ""
    model_used = None
    usage_dict = None
    gen_id = None

    if is_stream:
        # Handle streaming response
        async for chunk in completion:
            # Get model from first chunk
            if model_used is None and hasattr(chunk, 'model'):
                model_used = chunk.model
            if gen_id is None and hasattr(chunk, 'id'):
                gen_id = chunk.id
            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                resp_content += content
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                try:
                    usage_dict = _completion_usage_to_dict(chunk.usage)
                except Exception:
                    if logger:
                        logger.debug("Failed to convert streaming usage to dict", exc_info=True)
    else:
        # Handle non-streaming response
        if not completion.choices or len(completion.choices) == 0:
            raise ValueError("API returned empty choices list")
        resp_content = completion.choices[0].message.content
        model_used = completion.model if hasattr(completion, 'model') else None
        gen_id = getattr(completion, 'id', None)
        if hasattr(completion, 'usage') and completion.usage is not None:
            try:
                usage_dict = _completion_usage_to_dict(completion.usage)
            except Exception:
                if logger:
                    logger.debug("Failed to convert usage to dict", exc_info=True)

    openrouter_info = None
    if gen_id:
        openrouter_info = await _fetch_openrouter_generation_info(gen_id, logger)

    return resp_content.strip(), model_used, usage_dict, openrouter_info


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
    max_tokens: int = 32768,
):
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


def get_api_params_method(callId):
    return API_PARAMS_DICT.get(callId, lambda *args, **kwargs: {})


def make_openai_single_message(role: str, content: str):
    return {"role": role, "content": content}


def make_openai_message_system(content: str):
    return make_openai_single_message("system", content)


def make_openai_message_user(content: str):
    return make_openai_single_message("user", content)


def make_openai_message_assistant(content: str):
    return make_openai_single_message("assistant", content)


__all__ = [
    # core call helpers
    'openai_llm_call', 'async_openai_llm_call',
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
