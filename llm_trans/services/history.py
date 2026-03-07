from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm.history import ConversationTree
from llm.providers.multimodal import OpenRouterChatHistory, OpenRouterMessage

from llm.client.openai_compat import (
    make_openai_message_assistant,
    make_openai_message_system,
    make_openai_message_user,
)
from server.common.timezone import ensure_timezone, now_timestamp


@dataclass
class HistoryEvent:
    role: str
    content: str
    event_type: str
    attempt_number: int
    round_id: Optional[int] = None
    retry_index: Optional[int] = None
    reasoning_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: ensure_timezone(now_timestamp()).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if not payload.get("metadata"):
            payload["metadata"] = {}
        # Only include reasoning_content if it's not None and not empty
        if not payload.get("reasoning_content"):
            payload.pop("reasoning_content", None)
        return payload


class AttemptHistoryManager:
    """Manage conversation history for a single attempt using ConversationTree."""

    def __init__(
        self,
        *,
        attempt_work_dir: Path,
        attempt_number: int,
        case_type: str,
        case_name: str,
        resume: bool,
        logger,
    ) -> None:
        self.attempt_work_dir = attempt_work_dir
        self.attempt_number = attempt_number
        self.case_type = case_type
        self.case_name = case_name
        self.resume = resume
        self.logger = logger

        self.history_dir = attempt_work_dir / "logs" / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.history_dir / "conversation.json"

        self._tree: Optional[ConversationTree] = None
        self._conversation: List[Dict[str, Any]] = []
        self._events: List[HistoryEvent] = []
        self._system_prompt: Optional[str] = None

        if self.resume and self.history_file.exists():
            self._load_existing_history()

    @property
    def conversation(self) -> List[Dict[str, Any]]:
        return self._conversation

    def ensure_system_prompt(self, system_prompt: str) -> None:
        if self._system_prompt is not None:
            return
        self._system_prompt = system_prompt
        self._tree = ConversationTree(
            OpenRouterMessage,
            OpenRouterChatHistory,
            system_prompt=system_prompt,
            logger=self.logger,
        )
        system_message = make_openai_message_system(system_prompt)
        self._conversation.append(system_message)
        self._events.append(
            HistoryEvent(
                role="system",
                content=system_prompt,
                event_type="system",
                attempt_number=self.attempt_number,
                metadata={
                    "case_type": self.case_type,
                    "case_name": self.case_name,
                },
            )
        )
        self._save()

    def ensure_initial_user_prompt(self, content: str) -> None:
        if any(event.event_type == "user" for event in self._events):
            return
        self.add_user_message(
            content,
            round_id=1,
            retry_index=0,
            metadata={"kind": "initial_prompt"},
        )

    def add_user_message(
        self,
        content: str,
        *,
        round_id: Optional[int],
        retry_index: Optional[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._tree is None:
            raise RuntimeError("System prompt must be set before adding user messages")
        message = make_openai_message_user(content)
        self._conversation.append(message)
        self._tree.add_user_message(text=content)
        self._events.append(
            HistoryEvent(
                role="user",
                content=content,
                event_type="user",
                attempt_number=self.attempt_number,
                round_id=round_id,
                retry_index=retry_index,
                metadata=metadata or {},
            )
        )
        self._save()
        return message

    def add_assistant_message(
        self,
        reasoning_content: str,
        content: str,
        *,
        round_id: Optional[int],
        retry_index: Optional[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._tree is None:
            raise RuntimeError("System prompt must be set before adding assistant messages")
        # Only add content (without reasoning) to conversation for feedback context
        message = make_openai_message_assistant(content)
        self._conversation.append(message)
        self._tree.add_assistant_message(text=content)
        # Save both reasoning_content and content to events for logging
        self._events.append(
            HistoryEvent(
                role="assistant",
                content=content,
                event_type="assistant",
                attempt_number=self.attempt_number,
                round_id=round_id,
                retry_index=retry_index,
                reasoning_content=reasoning_content if reasoning_content else None,
                metadata=metadata or {},
            )
        )
        self._save()
        return message

    def add_error_event(
        self,
        *,
        stage: str,
        round_id: Optional[int],
        retry_index: Optional[int],
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._events.append(
            HistoryEvent(
                role="system",
                content=error,
                event_type="error",
                attempt_number=self.attempt_number,
                round_id=round_id,
                retry_index=retry_index,
                metadata={"stage": stage, **(metadata or {})},
            )
        )
        self._save()

    def _save(self) -> None:
        payload = [event.to_dict() for event in self._events]
        with open(self.history_file, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)

    def _load_existing_history(self) -> None:
        try:
            with open(self.history_file, "r", encoding="utf-8") as fp:
                raw_events = json.load(fp)
        except Exception as exc:  # pragma: no cover - corrupted history
            self.logger.warning(f"Failed to load history file {self.history_file}: {exc}")
            return

        events: List[HistoryEvent] = []
        for item in raw_events:
            event = HistoryEvent(
                role=item.get("role", "system"),
                content=item.get("content", ""),
                event_type=item.get("event_type", "system"),
                attempt_number=item.get("attempt_number", self.attempt_number),
                round_id=item.get("round_id"),
                retry_index=item.get("retry_index"),
                reasoning_content=item.get("reasoning_content"),
                metadata=item.get("metadata", {}),
                timestamp=item.get("timestamp", ensure_timezone(now_timestamp()).isoformat()),
            )
            events.append(event)

        system_event = next((ev for ev in events if ev.event_type == "system"), None)
        if not system_event:
            return

        self._system_prompt = system_event.content
        self._tree = ConversationTree(
            OpenRouterMessage,
            OpenRouterChatHistory,
            system_prompt=self._system_prompt,
            logger=self.logger,
        )

        self._events = []
        self._conversation = []

        for event in events:
            if event.event_type == "system":
                self._conversation.append(make_openai_message_system(event.content))
            elif event.event_type == "user":
                self._tree.add_user_message(text=event.content)
                self._conversation.append(make_openai_message_user(event.content))
            elif event.event_type == "assistant":
                self._tree.add_assistant_message(text=event.content)
                self._conversation.append(make_openai_message_assistant(event.content))
            # error events are skipped for conversation reconstruction
            self._events.append(event)

    def has_assistant_round(self, round_id: int) -> bool:
        return any(
            event.event_type == "assistant" and event.round_id == round_id
            for event in self._events
        )

    def next_round_id(self) -> int:
        round_ids = [
            event.round_id
            for event in self._events
            if event.event_type == "assistant" and event.round_id is not None
        ]
        return max(round_ids) + 1 if round_ids else 1
    def conversation_metadata(self) -> Dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "case_type": self.case_type,
            "case_name": self.case_name,
            "message_count": len(self._conversation),
        }


__all__ = ["AttemptHistoryManager", "HistoryEvent"]
