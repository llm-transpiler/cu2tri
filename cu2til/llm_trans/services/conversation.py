from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping

from utils.timezone import ensure_timezone, now_timestamp

from ..core.runtime import RuntimeContext


THINKING_PATTERNS = [
    r"<thought>(.*?)</thought>",
    r"<thinking>(.*?)</thinking>",
    r"<think>(.*?)</think>",
]


def extract_thinking_content(full_response: str) -> str | None:
    for pattern in THINKING_PATTERNS:
        match = re.search(pattern, full_response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def save_conversation_history(
    context: RuntimeContext,
    conversation_history: Iterable[Mapping[str, object]],
    test_work_dir: Path,
    *,
    round_num: int | None = None,
    timestamp: str | None = None,
) -> str | None:
    try:
        logs_dir = test_work_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        if round_num is not None:
            conversation_file = logs_dir / f"conversation_history_round_{round_num}.json"
        else:
            ts = timestamp or context.settings.timestamp
            conversation_file = logs_dir / f"conversation_history_{ts}.json"

        with open(conversation_file, "w", encoding="utf-8") as fp:
            json.dump(list(conversation_history), fp, indent=2, ensure_ascii=False)

        context.logger.debug(f"Conversation history saved to {conversation_file}")
        return str(conversation_file)
    except Exception as exc:  # pragma: no cover
        context.logger.warning(f"Failed to save conversation history: {exc}")
        return None


async def save_llm_conversation(
    context: RuntimeContext,
    test_work_dir: Path,
    attempt_number: int,
    round_id: int,
    retry_index: int,
    messages: list[dict],
    full_response: str,
    *,
    model_used: str | None = None,
    reasoning_content: str | None = None,
) -> None:
    conversations_dir = test_work_dir / "logs" / "conversations"
    conversations_dir.mkdir(parents=True, exist_ok=True)
    conversation_file = conversations_dir / "all_conversations.jsonl"

    timestamp = ensure_timezone(now_timestamp()).isoformat()

    try:
        with open(conversation_file, "a", encoding="utf-8") as fp:
            # 只保存当前轮次的最后一条 user 消息（如果存在）
            # 避免重复保存整个对话历史
            last_user_msg = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_msg = msg
                    break
            
            if last_user_msg:
                user_entry = {
                    "attempt_number": attempt_number,
                    "round_number": round_id,
                    "retry_index": retry_index,
                    "timestamp": timestamp,
                    "interaction_type": "request",
                    "role": "user",
                    "content": last_user_msg.get("content", ""),
                }
                fp.write(json.dumps(user_entry, ensure_ascii=False) + "\n")

            # 保存当前的 assistant response，包含 reasoning_content
            thinking_content = extract_thinking_content(full_response)

            response_entry = {
                "attempt_number": attempt_number,
                "round_number": round_id,
                "retry_index": retry_index,
                "timestamp": timestamp,
                "interaction_type": "response",
                "role": "assistant",
                "content": full_response,
                "success": True,
            }
            if model_used:
                response_entry["model_used"] = model_used
            if reasoning_content:
                response_entry["reasoning_content"] = reasoning_content
            if thinking_content:
                response_entry["thinking"] = thinking_content
                response_entry["content_without_thinking"] = re.sub(
                    r"<thought>.*?</thought>|<thinking>.*?</thinking>|<think>.*?</think>",
                    "",
                    full_response,
                    flags=re.DOTALL | re.IGNORECASE,
                ).strip()
            fp.write(json.dumps(response_entry, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover
        context.logger.warning(f"Failed to persist conversation: {exc}")


def get_last_code_block(resp_content: str, pl_tag: str = "python") -> str:
    resp_content = re.sub(r"<thought>.*?</thought>", "", resp_content, flags=re.DOTALL)

    python_starts: list[int] = []
    for match in re.finditer(f"```{pl_tag}\b", resp_content, re.IGNORECASE):
        python_starts.append(match.end())

    if python_starts:
        last_python_start = python_starts[-1]
        remaining = resp_content[last_python_start:]
        end_match = re.search(r"```", remaining)
        if end_match:
            return remaining[: end_match.start()].strip()
        return remaining.strip()

    code_block_pattern = r"```(\w+)?\s*(.*?)\s*```"
    matches = re.findall(code_block_pattern, resp_content, re.DOTALL)
    if matches:
        return matches[-1][1].strip()

    return resp_content


__all__ = [
    "extract_thinking_content",
    "save_conversation_history",
    "save_llm_conversation",
    "get_last_code_block",
]
