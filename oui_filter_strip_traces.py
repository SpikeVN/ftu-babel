"""
title: Strip Tool & Thinking Traces
description: Folds prior turns into a concise history block, strips past tool/thinking traces from context, and cleans stray closing think or tool_call leaks on outgoing replies.
author: FTU Labs
version: 0.3.0
"""

from __future__ import annotations

import re
from typing import Any, Optional

_TOOL_DETAILS_RE = re.compile(
    r'<details\b[^>]*type=["\']tool_calls["\'][^>]*>.*?</details>',
    flags=re.IGNORECASE | re.DOTALL,
)
_REASONING_DETAILS_RE = re.compile(
    r'<details\b[^>]*type=["\']reasoning["\'][^>]*>.*?</details>',
    flags=re.IGNORECASE | re.DOTALL,
)
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_ORPHAN_THINK_END_RE = re.compile(r"^.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)
_RAW_TOOL_CALL_RE = re.compile(
    r"<tool_call>.*?(?:</tool_call>|$)",
    flags=re.IGNORECASE | re.DOTALL,
)

_HISTORY_HEADER = "[Conversation history (earlier tool traces omitted)]"
_MAX_TURN_CHARS = 1500
_MAX_PRIOR_TURNS = 10


def _strip_traces_from_text(text: str) -> str:
    text = _TOOL_DETAILS_RE.sub("", text)
    text = _REASONING_DETAILS_RE.sub("", text)
    text = _THINK_RE.sub("", text)
    text = _ORPHAN_THINK_END_RE.sub("", text)
    text = _RAW_TOOL_CALL_RE.sub("", text)
    return text.strip()


def _extract_cleaned_text(content: Any) -> str:
    if isinstance(content, str):
        return _strip_traces_from_text(content)
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                t = _strip_traces_from_text(part.get("text", ""))
                if t:
                    chunks.append(t)
        return "\n".join(chunks).strip()
    return ""


def _clean_assistant_message(msg: dict[str, Any], *, drop_metadata: bool) -> dict[str, Any] | None:
    new_msg = dict(msg)
    if drop_metadata:
        new_msg.pop("tool_calls", None)
        new_msg.pop("reasoning", None)
        new_msg.pop("reasoning_content", None)

    content = new_msg.get("content")
    if isinstance(content, str):
        cleaned = _strip_traces_from_text(content)
        if not cleaned and drop_metadata:
            return None
        new_msg["content"] = cleaned
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                t = _strip_traces_from_text(part.get("text", ""))
                if t:
                    parts.append({**part, "text": t})
            else:
                parts.append(part)
        if not parts and drop_metadata:
            return None
        new_msg["content"] = parts
    return new_msg


class Filter:
    def inlet(self, body: dict[str, Any], __user__: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return body

        # Find the last user message index so current-turn tool calls remain intact
        last_user_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                last_user_idx = idx
                break

        if last_user_idx < 0:
            return body

        system_msgs: list[dict[str, Any]] = []
        prior_lines: list[str] = []
        for msg in messages[:last_user_idx]:
            role = msg.get("role")
            if role == "system":
                system_msgs.append(msg)
                continue
            if role == "tool":
                continue
            if role in ("user", "assistant"):
                if role == "assistant":
                    clean_msg = _clean_assistant_message(msg, drop_metadata=True)
                    if clean_msg is None:
                        continue
                    text = _extract_cleaned_text(clean_msg.get("content"))
                else:
                    text = _extract_cleaned_text(msg.get("content"))
                if not text:
                    continue
                if len(text) > _MAX_TURN_CHARS:
                    text = text[:_MAX_TURN_CHARS].rstrip() + "..."
                label = "User" if role == "user" else "Assistant"
                prior_lines.append(f"- {label}: {text}")

        if not prior_lines:
            return body

        prior_lines = prior_lines[-_MAX_PRIOR_TURNS:]
        history_prefix = (
            f"{_HISTORY_HEADER}\n"
            + "\n".join(prior_lines)
            + "\n\n[Current user message]\n"
        )

        last_user = dict(messages[last_user_idx])
        cur_content = last_user.get("content")
        if isinstance(cur_content, str):
            if not cur_content.startswith("[Conversation history ("):
                last_user["content"] = history_prefix + cur_content
        elif isinstance(cur_content, list):
            already_folded = any(
                isinstance(p, dict)
                and p.get("type") == "text"
                and str(p.get("text", "")).startswith("[Conversation history (")
                for p in cur_content
            )
            if not already_folded:
                last_user["content"] = [{"type": "text", "text": history_prefix}] + list(cur_content)

        body["messages"] = system_msgs + [last_user] + list(messages[last_user_idx + 1 :])
        return body

    def outlet(self, body: dict[str, Any], __user__: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        messages = body.get("messages")
        if isinstance(messages, list) and messages:
            updated: list[dict[str, Any]] = []
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant" and not msg.get("tool_calls"):
                    clean_msg = _clean_assistant_message(msg, drop_metadata=False)
                    updated.append(clean_msg if clean_msg is not None else msg)
                else:
                    updated.append(msg)
            body["messages"] = updated
        return body
