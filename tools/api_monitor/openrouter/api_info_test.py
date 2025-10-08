# https://openrouter.ai/docs/use-cases/usage-accounting
from llm.client.openai_compat import get_api_params_method, CallingIdentifier, make_openai_message_user
from pprint import pprint
import json
import time
import os
import asyncio
from openai import AsyncOpenAI
import dotenv
dotenv.load_dotenv()

from stats import *
from utils.util import obj_to_dict as _completion_usage_to_dict
from tools.api_monitor.openrouter import query as or_query

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# response = requests.get(
#   url="https://openrouter.ai/api/v1/key",
#   headers={
#     "Authorization": f"Bearer {OPENROUTER_API_KEY}"
#   }
# )

# print(json.dumps(response.json(), indent=2))

model_name = "anthropic/claude-sonnet-4"
model_name = "deepseek/deepseek-r1"
model_name = "openai/gpt-5-mini"
# model_name = "openrouter/auto"
# model_name = "openai/gpt-oss-120b"

messages = [make_openai_message_user("Which is bigger, 9.9 or 9.11?")]

get_api_param = get_api_params_method(CallingIdentifier.OPENAI_OPENROUTER)


async def run_single_request(client: AsyncOpenAI, stream: bool):
    api_params = get_api_param(messages, model_name)
    api_params['stream'] = stream
    # Fill max_tokens from OpenRouter query (output max)
    try:
        row = or_query.query_by_exact_name(model_name)
        if row and isinstance(row, dict):
            max_output = row.get('max_output')
            if isinstance(max_output, int) and max_output > 0:
                api_params['max_tokens'] = max_output
    except Exception:
        pass
    

    if not stream:
        pprint(api_params)
        _t0 = time.perf_counter()
        completion = await client.chat.completions.create(**api_params)
        _t1 = time.perf_counter()
        manual_req_e2e_ms = int((_t1 - _t0) * 1000)

        c_usage = getattr(completion, 'usage', None)
        c_usage_dict = _completion_usage_to_dict(c_usage)

        # Collect finish_reason from completion if available
        finish_reason_collected = None
        try:
            if completion and completion.choices and getattr(completion.choices[0], 'finish_reason', None):
                finish_reason_collected = completion.choices[0].finish_reason
        except Exception:
            pass

        report = await asyncio.to_thread(
            report_stats,
            completion.id,
            completion.model,
            manual_req_e2e_ms,
            None,
            c_usage_dict,
            finish_reason_collected,
        )
        print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
        est = await asyncio.to_thread(
            estimate_cost,
            model_name,
            c_usage_dict.get('prompt_tokens'),
            c_usage_dict.get('completion_tokens'),
        )
        print(est)
        print(completion.choices[0].message.content)
        return

    # stream path
    _t0 = time.perf_counter()
    gen_id = None
    model_used = None
    finish_reason = None
    manual_ttft_ms = None

    c_usage_dict = None
    reasoning_content = ""
    content = ""

    stream_resp = await client.chat.completions.create(**api_params)
    async for chunk in stream_resp:
        choices = getattr(chunk, 'choices', None) or []
        first_choice = choices[0] if len(choices) > 0 else None

        if gen_id is None:
            gen_id = getattr(chunk, 'id', None)
            model_used = getattr(chunk, 'model', None)

        try:
            if first_choice and hasattr(first_choice.delta, 'reasoning') and first_choice.delta.reasoning:
                reasoning_content += first_choice.delta.reasoning
                if manual_ttft_ms is None:
                    manual_ttft_ms = int((time.perf_counter() - _t0) * 1000)
            elif first_choice and first_choice.delta and first_choice.delta.content:
                content += first_choice.delta.content
                if manual_ttft_ms is None:
                    manual_ttft_ms = int((time.perf_counter() - _t0) * 1000)
        except Exception:
            print(f"Error: {Exception}")
            print(f"first_choice: {first_choice}")
            print(f"chunk: {chunk}")

        try:
            if first_choice and getattr(first_choice, 'finish_reason', None):
                finish_reason = first_choice.finish_reason
        except Exception:
            print(f"Error: {Exception}")
            print(f"first_choice: {first_choice}")
            print(f"chunk: {chunk}")

        if hasattr(chunk, 'usage') and chunk.usage is not None:
            try:
                c_usage_dict = _completion_usage_to_dict(chunk.usage)
            except Exception:
                print(f"Error: {Exception}")
                print(f"chunk: {chunk}")

    manual_req_e2e_ms = int((time.perf_counter() - _t0) * 1000)
    report = await asyncio.to_thread(
        report_stats,
        gen_id,
        (model_used or model_name),
        manual_req_e2e_ms,
        manual_ttft_ms,
        c_usage_dict,
        finish_reason,
    )
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    est = await asyncio.to_thread(
        estimate_cost,
        model_name,
        c_usage_dict.get('prompt_tokens') if c_usage_dict else None,
        c_usage_dict.get('completion_tokens') if c_usage_dict else None,
    )
    print(est)
    print(content)


async def main():
    async with AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    ) as client:
        # non-stream: run 3 concurrent requests
        await asyncio.gather(*[run_single_request(client, stream=False) for _ in range(1)])
        # stream: run 3 concurrent requests
        await asyncio.gather(*[run_single_request(client, stream=True) for _ in range(1)])


if __name__ == "__main__":
    asyncio.run(main())
'''
curl -G https://openrouter.ai/api/v1/generation \
     -H "Authorization: Bearer {OPENROUTER_API_KEY}" \
     -d id={gen_id}

看起来latency是被包含在generation_time中的
'''