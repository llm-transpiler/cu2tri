from openai import OpenAI
from enum import Enum
from logging import Logger


def openai_llm_call(client: OpenAI, api_params={}, logger: Logger = None):
    is_stream = api_params.get('stream', False)
    completion = client.chat.completions.create(**api_params)
    resp_content = ""
    if is_stream:
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                resp_content += content
    else:
        resp_content = completion.choices[0].message.content
    return resp_content.strip()


class CallingIdentifier(Enum):
    DEEPSEEK_OPENAI = 'deepseek_openai'
    CLAUDE_OPENROUTER = 'claude_openrouter'
    OPENAI_OFFICIAL = 'openai_official'
    GEMINI_OPENAI = 'gemini_openai'
    ANTHROPIC_OFFICIAL = 'anthropic_official'
    ANTHROPIC_OPENROUTER = 'anthropic_openrouter'
    OPENAI_OPENROUTER = 'openai_openrouter'


def get_api_param_gemini_openai(messages: list, model_name: str = "gemini-2.5-pro", stream: bool = True, temperature: float = 0.35, max_tokens: int = 65536, thinking: bool = True, thinking_budget: int = -1, include_thoughts: bool = False):
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
                        "include_thoughts": include_thoughts
                    }
                }
            }}
    return api_param

def get_api_param_openai_default(messages: list, model_name: str = "gpt-4o", stream: bool = True, temperature: float = 0.35, max_tokens: int = 65536):
    return {
        'model': model_name,
        'messages': messages,
        'stream': stream,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }

def get_api_param_deepseek_openai(messages: list, model_name: str = "deepseek-reasoner", stream: bool = True, temperature: float = 0.35, max_tokens: int = 65536):
    # 默认有thinking, 不需要额外参数(官方说法)
    return {
        'model': model_name,
        'messages': messages,
        'stream': stream,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }

def get_api_param_anthropic_openrouter(messages: list, model_name: str = "anthropic/claude-4-sonnet", stream: bool = True, temperature: float = 0.35, max_tokens: int = 65536, reasoning_effort: str = "high"):
    return {
        'model': model_name,
        'messages': messages,
        'stream': stream,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'reasoning_effort': reasoning_effort,
    }

def get_api_param_openai_openrouter(messages: list, model_name: str = "openai/gpt-5", stream: bool = True, temperature: float = 0.35, max_tokens: int = 65536, reasoning_effort: str = "high"):
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


def get_api_params_method(callId): return API_PARAMS_DICT.get(
    callId, lambda *args, **kwargs: {})

def make_openai_single_message(role: str, content: str):
    return {"role": role, "content": content}

def make_openai_message_system(content: str):
    return make_openai_single_message("system", content)

def make_openai_message_user(content: str):
    return make_openai_single_message("user", content)

def make_openai_message_assistant(content: str):
    return make_openai_single_message("assistant", content)