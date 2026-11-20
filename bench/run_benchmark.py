#!/usr/bin/env python3
"""
Benchmark runner for Genie Alpha on FTU500 benchmark.
Executes active multi-turn tool calling with Babel knowledge base (https://babel.falcolabs.org)
and evaluates model responses using DeepSeek-V4-Flash-0731 on DeepInfra with
Float Multi-Factor Scoring (factual accuracy, groundedness, completeness, adherence).
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import inspect
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import httpx
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# Ensure workspace root is in path and .env is loaded
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from oui_filter_strip_traces import Filter as StripTracesFilter
from oui_tool import Tools as BabelToolHandler

OPENWEBUI_ENDPOINT = os.getenv("OPENWEBUI_ENDPOINT", "").rstrip("/")
OPENWEBUI_KEY = os.getenv("OPENWEBUI_KEY", "")
DEEPINFRA_AUTH_TOKEN = os.getenv("DEEPINFRA_AUTH_TOKEN", "")

DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEFAULT_MODEL = "genie-01-alpha"
DEFAULT_JUDGE_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_TOOL_IDS = ["ftu_babel_query"]
ALLOWED_TOOL_NAMES = {"babel_query", "babel_fetch", "web_search"}
TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}
VALID_VERDICTS = {"CORRECT", "PARTIAL", "INCORRECT"}

ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[94m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"


def format_colored_log(text: str, color: str) -> str:
    """Wrap text in ANSI color codes unless NO_COLOR is set."""
    if os.getenv("NO_COLOR"):
        return text
    return f"{color}{text}{ANSI_RESET}"


def compact_preview(text: str, max_len: int = 240) -> str:
    """Normalize whitespace and truncate text for readable single-line terminal logs."""
    single = " ".join((text or "").split())
    if len(single) <= max_len:
        return single
    return single[: max_len - 1] + "…"

# Load the official Genie system prompt (used when --use-local-prompt is enabled)
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "SYSTEM_PROMPT.md"
if SYSTEM_PROMPT_PATH.exists():
    SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
else:
    SYSTEM_PROMPT = (
        "Genie is the assistant for FTU (Trường Đại học Ngoại thương). "
        "Answers FTU facts using only tool results retrieved via babel_query and babel_fetch."
    )

_DOC_ID_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {"type": "integer"},
        {"type": "string"},
        {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                ]
            },
        },
    ]
}

# Fallback OpenAI tool specs aligned with oui_tool.py (used when --use-local-prompt is set)
BABEL_TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "babel_query",
            "description": (
                "Primary mandatory search tool for the FTU (Trường Đại học Ngoại thương) knowledge base. "
                "ALWAYS call this before web_search for any FTU or Vietnamese student/university question. "
                "Returns matched passages with doc_id and Chunk coordinates for targeted babel_fetch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string (use user's literal wording, typos included).",
                    },
                    "mode": {
                        "type": "string",
                        "default": "semantic",
                        "enum": ["semantic", "keyword", "hybrid"],
                        "description": (
                            "Search mode: 'semantic' (default), 'keyword' (BM25 for names/codes/acronyms), "
                            "or 'hybrid'."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of passage results to return (1-100, default 10).",
                    },
                    "format": {
                        "type": "string",
                        "default": "text",
                        "enum": ["text", "json"],
                        "description": "Output format, either 'text' (default) or 'json'.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "babel_fetch",
            "description": (
                "Fetch document content and metadata from Babel. Short documents return in full automatically. "
                "For long documents, pass chunk_index (from babel_query's Chunk: X of 0..N) plus window (default 1) "
                "to retrieve the target section and adjacent neighbor chunks without context overflow."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_ids": {
                        **_DOC_ID_SCHEMA,
                        "description": "A document ID (e.g., 1525) or list of doc_ids returned by babel_query.",
                    },
                    "doc_id": {
                        **_DOC_ID_SCHEMA,
                        "description": "Optional alias for a single document ID or list of doc_ids.",
                    },
                    "ids": {
                        **_DOC_ID_SCHEMA,
                        "description": "Optional alias for doc_ids.",
                    },
                    "chunk_index": {
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                        "description": "Optional 0-based chunk index from babel_query's Chunk: X of 0..N line.",
                    },
                    "window": {
                        "type": "integer",
                        "default": 1,
                        "description": "Number of neighboring chunks before and after chunk_index to include (0-5, default 1).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Optional per-document character cap override (1000-100000).",
                    },
                    "clean": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to strip links, images, and headers so only normal text remains.",
                    },
                    "format": {
                        "type": "string",
                        "default": "text",
                        "enum": ["text", "json"],
                        "description": "Output format, either 'text' (default) or 'json'.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "FALLBACK ONLY — NEVER call web_search before babel_query for FTU or Vietnamese university questions. "
                "Search the public web via SearXNG only after babel_query / babel_fetch have run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query string.",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of search results to return (1-20, default 5).",
                    },
                    "format": {
                        "type": "string",
                        "default": "text",
                        "enum": ["text", "json"],
                        "description": "Output format, either 'text' (default) or 'json'.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

JUDGE_SYSTEM_PROMPT = """Bạn là giám khảo LLM (LLM Judge) chuyên nghiệp, công tâm, khách quan và nghiêm ngặt.
Nhiệm vụ của bạn là đánh giá câu trả lời của Trợ lý AI (Model Response) dựa trên Câu hỏi, Đáp án tham chiếu (Ground Truth), Hành vi kỳ vọng (Expected Behavior) và Hướng dẫn chấm (Grading Notes) theo thang điểm đa nhân tố dạng số thực (Float Multi-Factor Scoring) từ 0.00 đến 1.00.

Các nhân tố đánh giá (0.00 đến 1.00):
1. factual_accuracy (0.00 - 1.00): Tính chính xác của các dữ kiện cốt lõi (con số, ngày tháng, tên người, cơ sở, sự kiện, chính sách). 1.0 = hoàn toàn chuẩn xác; 0.0 = sai lệch dữ kiện chính; 0.5 = đúng được một nửa ý chính.
2. groundedness (0.00 - 1.00): Mức độ trung thực và phi bịa đặt (hallucination-free). 1.0 = không có bất kỳ thông tin bịa đặt nào; 0.0 = tự bịa đặt trắng trợn các dữ kiện không có thật.
3. completeness (0.00 - 1.00): Độ bao quát đầy đủ các ý cốt lõi theo Ground Truth và Grading Notes.
4. adherence (0.00 - 1.00): Tuân thủ chặt chẽ Hành vi kỳ vọng (Expected Behavior: 'answer', 'refuse_no_info', 'correct_premise', 'out_of_scope'):
   - 'answer': phải cung cấp câu trả lời rõ ràng.
   - 'refuse_no_info': phải thừa nhận không có/chưa có thông tin hoặc đề nghị kiểm tra kênh chính thống; không được bịa chi tiết.
   - 'correct_premise': phải chỉ ra tiền đề sai và đính chính theo thực tế; không được hùa theo tiền đề sai.
   - 'out_of_scope': phải từ chối lịch sự hoặc chuyển hướng phù hợp.

Công thức tính điểm tổng hợp:
final_score = round(0.45 * factual_accuracy + 0.25 * groundedness + 0.15 * completeness + 0.15 * adherence, 2)

Phân loại verdict:
- "CORRECT": final_score >= 0.80
- "PARTIAL": 0.40 <= final_score < 0.80
- "INCORRECT": final_score < 0.40

Định dạng đầu ra BẮT BUỘC là JSON duy nhất:
{
  "factual_accuracy": float,
  "groundedness": float,
  "completeness": float,
  "adherence": float,
  "final_score": float,
  "verdict": "CORRECT" | "PARTIAL" | "INCORRECT",
  "reason": "Giải thích ngắn gọn 1-2 câu lý do chấm điểm và nhận xét về dữ kiện."
}"""


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_ORPHAN_THINK_END_RE = re.compile(r"^.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)
_DETAILS_TRACE_RE = re.compile(
    r'<details\b[^>]*type=["\'](?:reasoning|tool_calls)["\'][^>]*>.*?</details>',
    flags=re.IGNORECASE | re.DOTALL,
)
_RAW_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>.*?(?:</tool_call>|$)",
    flags=re.IGNORECASE | re.DOTALL,
)
_RAW_TOOL_MARKER_RE = re.compile(r"<tool_call\b|<function=", flags=re.IGNORECASE)


def sanitize_model_response(text: str) -> str:
    """Strip leaked thinking traces, reasoning/tool details blocks, and raw XML tool_call tags."""
    if not text:
        return ""
    cleaned = _DETAILS_TRACE_RE.sub("", text)
    cleaned = _THINK_BLOCK_RE.sub("", cleaned)
    cleaned = _ORPHAN_THINK_END_RE.sub("", cleaned)
    cleaned = _RAW_TOOL_CALL_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def _contains_raw_tool_call(text: str) -> bool:
    """Return True if text still contains raw XML tool_call or function markers."""
    return bool(_RAW_TOOL_MARKER_RE.search(text or ""))


def strip_json_fences(text: str) -> str:
    """Strip optional markdown code fences around JSON output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def clamp01(value: Any) -> float:
    """Coerce a numeric factor to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def score_and_verdict(raw_content: str) -> dict[str, Any]:
    """Parse judge JSON, clamp factors to [0, 1], and deterministically compute score & verdict."""
    parsed = json.loads(strip_json_fences(raw_content))
    if not isinstance(parsed, dict):
        raise ValueError("Judge response JSON must be an object.")

    factual = clamp01(parsed.get("factual_accuracy", 0.0))
    grounded = clamp01(parsed.get("groundedness", 0.0))
    complete = clamp01(parsed.get("completeness", 0.0))
    adhere = clamp01(parsed.get("adherence", 0.0))

    final_score = round(
        0.45 * factual + 0.25 * grounded + 0.15 * complete + 0.15 * adhere,
        2,
    )
    if final_score >= 0.80:
        verdict = "CORRECT"
    elif final_score >= 0.40:
        verdict = "PARTIAL"
    else:
        verdict = "INCORRECT"

    return {
        "factual_accuracy": factual,
        "groundedness": grounded,
        "completeness": complete,
        "adherence": adhere,
        "final_score": final_score,
        "verdict": verdict,
        "reason": str(parsed.get("reason", "")).strip(),
        "error": None,
    }


def extract_user_messages(item: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Return conversation messages for the model and the latest question text."""
    conv = item.get("conversation")
    if isinstance(conv, list) and conv:
        q_text = (conv[-1].get("content") or item.get("question") or "").strip()
        return conv, q_text
    q_text = (item.get("question") or "").strip()
    return [{"role": "user", "content": q_text}], q_text


def format_question_for_judge(item: dict[str, Any]) -> str:
    """Format single-turn or multi-turn question with prior context for the LLM judge."""
    conv = item.get("conversation")
    if isinstance(conv, list) and len(conv) > 1:
        history_lines: list[str] = []
        for turn in conv[:-1]:
            role_label = "Người dùng" if turn.get("role") == "user" else "Trợ lý"
            content = (turn.get("content") or "").strip()
            history_lines.append(f"{role_label}: {content}")
        last_q = (conv[-1].get("content") or item.get("question") or "").strip()
        return (
            "[Hội thoại trước đó]\n"
            + "\n".join(history_lines)
            + f"\n[Câu hỏi cần đánh giá]\n{last_q}"
        )
    if isinstance(conv, list) and len(conv) == 1:
        return (conv[0].get("content") or item.get("question") or "").strip()
    return (item.get("question") or "").strip()


async def post_json_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: float,
    max_retries: int = 3,
    retry_statuses: set[int] = TRANSIENT_HTTP_STATUSES,
) -> httpx.Response:
    """POST JSON with exponential backoff and jitter on transient HTTP or network errors."""
    last_exc: Exception | None = None
    last_resp: httpx.Response | None = None

    for attempt in range(max_retries):
        try:
            resp = await client.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code not in retry_statuses:
                return resp
            last_resp = resp
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc

        if attempt < max_retries - 1:
            delay = min(16.0, float(2 ** (attempt + 1))) + random.uniform(0.0, 0.5)
            await asyncio.sleep(delay)

    if last_resp is not None:
        return last_resp
    assert last_exc is not None
    raise last_exc


async def execute_tool_call(tool_handler: BabelToolHandler, name: str, args_str: Any) -> str:
    """Execute a Babel tool call properly whether the target method is async or sync."""
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
    except Exception as exc:
        return f"Error parsing tool arguments: {exc}"

    if not isinstance(args, dict):
        return f"Error: Tool arguments must be a JSON object, got {type(args).__name__}."

    clean_args = {k: v for k, v in args.items() if not str(k).startswith("__")}

    if name not in ALLOWED_TOOL_NAMES:
        return f"Error: Tool '{name}' not found."

    fn = getattr(tool_handler, name, None)
    if not callable(fn):
        return f"Error: Tool '{name}' not found."

    try:
        if inspect.iscoroutinefunction(fn):
            res = await fn(**clean_args)
        else:
            res = await asyncio.to_thread(fn, **clean_args)
            if inspect.isawaitable(res):
                res = await res
        return str(res)
    except Exception as exc:
        return f"Tool execution error: {exc}"


async def run_genie_agent_turn(
    client: httpx.AsyncClient,
    model: str,
    user_messages: list[dict[str, Any]],
    tool_handler: BabelToolHandler,
    max_tool_iterations: int = 8,
    timeout: float = 90.0,
    use_local_prompt: bool = False,
    tool_ids: list[str] | None = None,
    max_retries: int = 3,
    item_id: str | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Run a full Genie Alpha turn with multi-step parallel tool execution.
    Uses OpenWebUI's live prompt and tool_ids by default, or local prompt/specs if requested.
    """
    prefix = f"[{model}][{item_id}]" if item_id else f"[{model}]"

    def _log(text: str) -> None:
        if log_fn is not None:
            log_fn(format_colored_log(f"{prefix} {text}", ANSI_YELLOW))

    headers = {
        "Authorization": f"Bearer {OPENWEBUI_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{OPENWEBUI_ENDPOINT}/api/chat/completions"

    resolved_tool_ids = list(DEFAULT_TOOL_IDS) if tool_ids is None else list(tool_ids)

    if use_local_prompt:
        current_date = datetime.now().strftime("%d/%m/%Y")
        local_prompt = SYSTEM_PROMPT.replace("{{CURRENT_DATE}}", current_date)
        messages: list[dict[str, Any]] = [{"role": "system", "content": local_prompt}]
        messages.extend(user_messages)
        if len(user_messages) > 1:
            messages = StripTracesFilter().inlet({"messages": messages})["messages"]
        base_payload: dict[str, Any] = {
            "model": model,
            "tools": BABEL_TOOLS_SPEC,
            "stream": False,
        }
    else:
        messages = list(user_messages)
        base_payload = {
            "model": model,
            "stream": False,
        }
        if resolved_tool_ids:
            base_payload["tool_ids"] = resolved_tool_ids

    t0 = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    executed_tools: list[dict[str, Any]] = []

    last_user_msg = (user_messages[-1].get("content") or "") if user_messages else ""
    _log(f"Q: {compact_preview(last_user_msg, 160)}")

    def _error_result(err: str) -> dict[str, Any]:
        elapsed = time.time() - t0
        _log(f"ERR ({elapsed:.1f}s): {compact_preview(err, 200)}")
        return {
            "content": "",
            "latency": elapsed,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "tools_called": executed_tools,
            "error": err,
        }

    for step_idx in range(1, max_tool_iterations + 1):
        payload = {**base_payload, "messages": messages}
        try:
            r = await post_json_with_retry(
                client, url, headers, payload, timeout=timeout, max_retries=max_retries
            )
            if r.status_code != 200:
                return _error_result(f"OpenWebUI HTTP {r.status_code}: {r.text[:200]}")

            data = r.json()
            choice = data["choices"][0]
            msg = choice.get("message", {})
            usage = data.get("usage") or {}
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                raw_content = (msg.get("content") or "").strip()
                cleaned = sanitize_model_response(raw_content)
                if _contains_raw_tool_call(raw_content) and not cleaned:
                    _log(f"step {step_idx} emitted raw XML tool tag; falling back to synthesis")
                    break
                content = cleaned
                elapsed = time.time() - t0
                _log(
                    f"A ({elapsed:.1f}s, {len(executed_tools)} tools, "
                    f"{total_prompt_tokens}->{total_completion_tokens} tok): {compact_preview(content, 240)}"
                )
                return {
                    "content": content,
                    "latency": elapsed,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "tools_called": executed_tools,
                    "error": None,
                }

            messages.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            })

            pending_calls: list[tuple[dict[str, Any], str, Any]] = []
            for tc in tool_calls:
                fn_obj = tc.get("function") or {}
                fn_name = fn_obj.get("name", "")
                fn_args = fn_obj.get("arguments", "{}")
                executed_tools.append({"name": fn_name, "args": fn_args})
                pending_calls.append((tc, fn_name, fn_args))
                _log(f"step {step_idx} -> {fn_name}({compact_preview(str(fn_args), 140)})")

            async def _timed_tool(fn_name: str, fn_args: Any) -> tuple[str, float]:
                t_tool = time.time()
                out = await execute_tool_call(tool_handler, fn_name, fn_args)
                return out, time.time() - t_tool

            tool_outputs = await asyncio.gather(
                *(_timed_tool(fn_name, fn_args) for _, fn_name, fn_args in pending_calls)
            )
            for (tc, fn_name, _), (tool_result, tool_elapsed) in zip(pending_calls, tool_outputs):
                _log(f"step {step_idx} <- {fn_name} ({len(tool_result)} chars, {tool_elapsed:.1f}s)")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": tool_result,
                })
        except Exception as exc:
            return _error_result(str(exc))

    # Max tool iterations (or raw XML fallback) reached: perform one final synthesis call with tool_choice="none"
    final_messages = list(messages) + [
        {
            "role": "user",
            "content": (
                "[Hệ thống: Đã hết lượt gọi công cụ. Không gọi thêm công cụ hay xuất thẻ <tool_call>. "
                "Hãy trả lời trực tiếp câu hỏi của người dùng dựa trên các kết quả đã tra cứu được; "
                "nếu chưa tìm thấy thông tin thì nêu rõ không tìm thấy và hướng dẫn liên hệ phòng/ban phù hợp.]"
            ),
        }
    ]
    final_payload = {**base_payload, "messages": final_messages, "tool_choice": "none"}
    if use_local_prompt:
        final_payload.pop("tools", None)
    try:
        r = await post_json_with_retry(
            client, url, headers, final_payload, timeout=timeout, max_retries=max_retries
        )
        if r.status_code == 200:
            data = r.json()
            msg = data["choices"][0].get("message", {})
            usage = data.get("usage") or {}
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            raw_content = (msg.get("content") or "").strip()
            cleaned = sanitize_model_response(raw_content)
            if not (msg.get("tool_calls") or []) and cleaned and not _contains_raw_tool_call(cleaned):
                elapsed = time.time() - t0
                _log(
                    f"A ({elapsed:.1f}s, {len(executed_tools)} tools, "
                    f"{total_prompt_tokens}->{total_completion_tokens} tok): {compact_preview(cleaned, 240)}"
                )
                return {
                    "content": cleaned,
                    "latency": elapsed,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "tools_called": executed_tools,
                    "error": None,
                }
    except Exception:
        pass

    return _error_result("Max tool iterations reached")


async def call_judge_with_retry(
    client: httpx.AsyncClient,
    judge_model: str,
    question_text: str,
    answer_text: str,
    expected_behavior: str,
    grading_notes: str,
    model_response: str,
    max_retries: int = 5,
    timeout: float = 45.0,
) -> dict[str, Any]:
    """Call DeepInfra LLM Judge to evaluate model response with Float Multi-Factor Scoring."""
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_AUTH_TOKEN}",
        "Content-Type": "application/json",
    }

    user_prompt = f"""[CÂU HỎI]: {question_text}
[ĐÁP ÁN THAM CHIẾU]: {answer_text}
[HÀNH VI KỲ VỌNG]: {expected_behavior}
[HƯỚNG DẪN CHẤM]: {grading_notes}
[CÂU TRẢ LỜI CỦA MODEL]: {model_response}"""

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }

    last_err: Exception | str | None = None
    for attempt in range(max_retries):
        try:
            r = await client.post(DEEPINFRA_API_URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                raw_json = r.json()["choices"][0]["message"].get("content") or ""
                return score_and_verdict(raw_json)
            elif r.status_code in TRANSIENT_HTTP_STATUSES:
                last_err = f"Judge HTTP {r.status_code}: {r.text[:200]}"
            else:
                return {
                    "factual_accuracy": 0.0,
                    "groundedness": 0.0,
                    "completeness": 0.0,
                    "adherence": 0.0,
                    "final_score": 0.0,
                    "verdict": "ERROR",
                    "reason": f"Judge HTTP {r.status_code}: {r.text[:200]}",
                    "error": r.text,
                }
        except Exception as exc:
            last_err = exc

        if attempt < max_retries - 1:
            delay = min(16.0, float(2 ** (attempt + 1))) + random.uniform(0.0, 0.5)
            await asyncio.sleep(delay)

    return {
        "factual_accuracy": 0.0,
        "groundedness": 0.0,
        "completeness": 0.0,
        "adherence": 0.0,
        "final_score": 0.0,
        "verdict": "ERROR",
        "reason": f"Judge error: {last_err}",
        "error": str(last_err),
    }


async def evaluate_single_item(
    item: dict[str, Any],
    model: str,
    judge_model: str,
    client: httpx.AsyncClient,
    tool_handler: BabelToolHandler,
    sem: asyncio.Semaphore,
    *,
    use_local_prompt: bool = False,
    tool_ids: list[str] | None = None,
    max_tool_iterations: int = 8,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Process a single benchmark item through Genie tool turn and multi-factor judge evaluation."""
    async with sem:
        messages, q_text = extract_user_messages(item)
        judge_q_text = format_question_for_judge(item)

        # 1. Run Genie turn with Babel tools
        turn_res = await run_genie_agent_turn(
            client=client,
            model=model,
            user_messages=messages,
            tool_handler=tool_handler,
            max_tool_iterations=max_tool_iterations,
            use_local_prompt=use_local_prompt,
            tool_ids=tool_ids,
            item_id=item["id"],
            log_fn=log_fn,
        )
        model_output = turn_res["content"]

        # 2. Query LLM Judge
        if turn_res["error"] or not model_output:
            err_msg = turn_res["error"] or "Empty model response"
            judge_res = {
                "factual_accuracy": 0.0,
                "groundedness": 0.0,
                "completeness": 0.0,
                "adherence": 0.0,
                "final_score": 0.0,
                "verdict": "ERROR",
                "reason": f"Inference error: {err_msg}",
                "error": err_msg,
            }
        else:
            judge_res = await call_judge_with_retry(
                client=client,
                judge_model=judge_model,
                question_text=judge_q_text,
                answer_text=item.get("answer", ""),
                expected_behavior=item.get("expected_behavior", "answer"),
                grading_notes=item.get("grading_notes", ""),
                model_response=model_output,
            )

        if log_fn is not None:
            log_fn(
                format_colored_log(
                    f"[judge][{item['id']}] {judge_res['verdict']} ({judge_res['final_score']:.2f}) "
                    f"- {compact_preview(judge_res['reason'], 160)}",
                    ANSI_DIM,
                )
            )

        return {
            "id": item["id"],
            "type": item.get("type"),
            "subtype": item.get("subtype"),
            "difficulty": item.get("difficulty"),
            "category": item.get("category"),
            "question": q_text,
            "answer": item.get("answer", ""),
            "expected_behavior": item.get("expected_behavior", "answer"),
            "grading_notes": item.get("grading_notes", ""),
            "model_response": model_output,
            "tools_called": turn_res["tools_called"],
            "model_latency_sec": round(turn_res["latency"], 3),
            "prompt_tokens": turn_res["prompt_tokens"],
            "completion_tokens": turn_res["completion_tokens"],
            "factual_accuracy": judge_res["factual_accuracy"],
            "groundedness": judge_res["groundedness"],
            "completeness": judge_res["completeness"],
            "adherence": judge_res["adherence"],
            "judge_score": judge_res["final_score"],
            "judge_verdict": judge_res["verdict"],
            "judge_reason": judge_res["reason"],
            "error": turn_res.get("error") or judge_res.get("error"),
        }


def compute_summary_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute comprehensive accuracy and multi-factor breakdown statistics."""
    total = len(results)
    if total == 0:
        return {}

    avg_final = sum(r.get("judge_score", 0.0) for r in results) / total
    avg_factual = sum(r.get("factual_accuracy", 0.0) for r in results) / total
    avg_grounded = sum(r.get("groundedness", 0.0) for r in results) / total
    avg_complete = sum(r.get("completeness", 0.0) for r in results) / total
    avg_adhere = sum(r.get("adherence", 0.0) for r in results) / total

    verdict_counts: dict[str, int] = {}
    for r in results:
        v = r.get("judge_verdict", "UNKNOWN")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    def breakdown_by(field: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            val = r.get(field) or "Unspecified"
            groups.setdefault(val, []).append(r)

        breakdown: dict[str, Any] = {}
        for k, group in groups.items():
            g_total = len(group)
            g_score = sum(x.get("judge_score", 0.0) for x in group)
            breakdown[k] = {
                "count": g_total,
                "avg_final_score": round(g_score / g_total, 3),
                "avg_factual": round(sum(x.get("factual_accuracy", 0.0) for x in group) / g_total, 3),
                "avg_grounded": round(sum(x.get("groundedness", 0.0) for x in group) / g_total, 3),
                "correct": sum(1 for x in group if x.get("judge_verdict") == "CORRECT"),
                "partial": sum(1 for x in group if x.get("judge_verdict") == "PARTIAL"),
                "incorrect": sum(1 for x in group if x.get("judge_verdict") == "INCORRECT"),
                "error": sum(1 for x in group if x.get("judge_verdict") == "ERROR"),
            }
        return breakdown

    latencies = sorted(r["model_latency_sec"] for r in results if r.get("model_latency_sec"))
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p50_latency = latencies[min(len(latencies) - 1, int(len(latencies) * 0.50))] if latencies else 0.0
    p90_latency = latencies[min(len(latencies) - 1, int(len(latencies) * 0.90))] if latencies else 0.0

    total_tools_called = sum(len(r.get("tools_called") or []) for r in results)

    return {
        "total_questions": total,
        "multi_factor_averages": {
            "overall_final_score": round(avg_final, 3),
            "factual_accuracy": round(avg_factual, 3),
            "groundedness": round(avg_grounded, 3),
            "completeness": round(avg_complete, 3),
            "adherence": round(avg_adhere, 3),
        },
        "verdict_distribution": verdict_counts,
        "tools_usage": {
            "total_tools_called": total_tools_called,
            "avg_tools_per_question": round(total_tools_called / total, 2),
        },
        "by_type": breakdown_by("type"),
        "by_difficulty": breakdown_by("difficulty"),
        "by_expected_behavior": breakdown_by("expected_behavior"),
        "by_category": breakdown_by("category"),
        "latency_stats": {
            "avg_sec": round(avg_latency, 2),
            "p50_sec": round(p50_latency, 2),
            "p90_sec": round(p90_latency, 2),
        },
    }


def _is_completed_record(record: dict[str, Any]) -> bool:
    return (
        record.get("judge_verdict") in VALID_VERDICTS
        and not record.get("error")
        and "factual_accuracy" in record
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run FTU benchmark against Genie Alpha with Babel Tooling")
    parser.add_argument("--bench-file", default="bench/ftu500.json", help="Path to benchmark JSON file")
    parser.add_argument("--output-jsonl", default="bench/results_genie_alpha.jsonl", help="Output JSONL path")
    parser.add_argument("--output-summary", default="bench/summary_genie_alpha.json", help="Output summary JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Target model in OpenWebUI")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model on DeepInfra")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent requests")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit of questions to benchmark (0 = all)")
    parser.add_argument("--ids", default="", help="Comma-separated question IDs to run (e.g. Q001,MT-001)")
    parser.add_argument(
        "--use-local-prompt",
        action="store_true",
        help="Send local SYSTEM_PROMPT.md and BABEL_TOOLS_SPEC instead of OpenWebUI server-side prompt & tool_ids",
    )
    parser.add_argument(
        "--tool-ids",
        default=",".join(DEFAULT_TOOL_IDS),
        help="Comma-separated OpenWebUI tool IDs (used when --use-local-prompt is off)",
    )
    parser.add_argument("--max-tool-iterations", type=int, default=8, help="Max tool-calling rounds per question")
    parser.add_argument("--quiet", action="store_true", help="Disable live colored chat and tool logs")
    args = parser.parse_args()

    if not OPENWEBUI_ENDPOINT or not OPENWEBUI_KEY:
        print("Error: OPENWEBUI_ENDPOINT and OPENWEBUI_KEY must be set in .env or environment.")
        sys.exit(1)
    if not DEEPINFRA_AUTH_TOKEN:
        print("Error: DEEPINFRA_AUTH_TOKEN must be set in .env or environment.")
        sys.exit(1)

    bench_path = Path(args.bench_file)
    if not bench_path.exists():
        print(f"Error: {bench_path} not found.")
        sys.exit(1)

    with open(bench_path, "r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    if args.ids.strip():
        wanted_ids = {qid.strip() for qid in args.ids.split(",") if qid.strip()}
        data = [d for d in data if d["id"] in wanted_ids]

    if args.limit > 0:
        data = data[: args.limit]

    if not data:
        print("Error: No benchmark items matched the selection.")
        sys.exit(1)

    tool_ids = [t.strip() for t in args.tool_ids.split(",") if t.strip()]
    target_ids = {d["id"] for d in data}

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)

    records_by_id: dict[str, dict[str, Any]] = {}
    if output_jsonl.exists():
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "id" in record:
                        records_by_id[record["id"]] = record
                except Exception:
                    pass

    completed_by_id = {
        qid: rec for qid, rec in records_by_id.items()
        if qid in target_ids and _is_completed_record(rec)
    }
    completed_ids = set(completed_by_id.keys())
    if records_by_id:
        print(
            f"Resuming benchmark: Found {len(completed_ids)} already completed items "
            f"(out of {len(records_by_id)} records) in {output_jsonl}"
        )

    remaining_items = [d for d in data if d["id"] not in completed_ids]
    print(f"Total benchmark items: {len(data)} | Remaining: {len(remaining_items)}")

    running_by_id: dict[str, dict[str, Any]] = dict(completed_by_id)

    if remaining_items:
        sem = asyncio.Semaphore(args.concurrency)
        tool_handler = BabelToolHandler()
        file_mode = "a" if output_jsonl.exists() and records_by_id else "w"

        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(output_jsonl, file_mode, encoding="utf-8") as out_file:
                pbar = tqdm(
                    total=len(data),
                    initial=len(completed_ids),
                    desc=f"Benchmarking {args.model}",
                    dynamic_ncols=True,
                )
                log_fn = None if args.quiet else (lambda msg: pbar.write(msg, file=sys.stderr))

                async def worker(item: dict[str, Any]) -> None:
                    res = await evaluate_single_item(
                        item=item,
                        model=args.model,
                        judge_model=args.judge_model,
                        client=client,
                        tool_handler=tool_handler,
                        sem=sem,
                        use_local_prompt=args.use_local_prompt,
                        tool_ids=tool_ids,
                        max_tool_iterations=args.max_tool_iterations,
                        log_fn=log_fn,
                    )
                    out_file.write(json.dumps(res, ensure_ascii=False) + "\n")
                    out_file.flush()
                    running_by_id[res["id"]] = res
                    pbar.update(1)

                    cur_scores = [x["judge_score"] for x in running_by_id.values()]
                    cur_avg = sum(cur_scores) / len(cur_scores)
                    cur_errs = sum(
                        1 for x in running_by_id.values()
                        if x.get("judge_verdict") == "ERROR" or x.get("error")
                    )
                    pbar.set_postfix({
                        "avg_score": f"{cur_avg:.2f}",
                        "errs": cur_errs,
                        "tools": len(res.get("tools_called", [])),
                    })

                await asyncio.gather(*(worker(item) for item in remaining_items))
                pbar.close()

    ordered_results = [running_by_id[d["id"]] for d in data if d["id"] in running_by_id]

    print("\nComputing final benchmark summary...")
    summary = compute_summary_statistics(ordered_results)
    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n================ BENCHMARK COMPLETED ================")
    print(f"Model: {args.model}")
    print(f"Judge: {args.judge_model}")
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Overall Final Score: {summary['multi_factor_averages']['overall_final_score']}")
    print(f"Factual Accuracy: {summary['multi_factor_averages']['factual_accuracy']}")
    print(f"Groundedness: {summary['multi_factor_averages']['groundedness']}")
    print(f"Completeness: {summary['multi_factor_averages']['completeness']}")
    print(f"Adherence: {summary['multi_factor_averages']['adherence']}")
    print(f"Verdict Distribution: {summary['verdict_distribution']}")
    print(f"Total Tools Called: {summary['tools_usage']['total_tools_called']}")
    print(f"Summary saved to: {args.output_summary}")
    print(f"Details saved to: {args.output_jsonl}")


if __name__ == "__main__":
    asyncio.run(main())
