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
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# Ensure workspace root is in path and .env is loaded
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from oui_tool import Tools as BabelToolHandler

OPENWEBUI_ENDPOINT = os.getenv("OPENWEBUI_ENDPOINT", "").rstrip("/")
OPENWEBUI_KEY = os.getenv("OPENWEBUI_KEY", "")
DEEPINFRA_AUTH_TOKEN = os.getenv("DEEPINFRA_AUTH_TOKEN", "")

DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEFAULT_MODEL = "genie-01-alpha"
DEFAULT_JUDGE_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

# Load the official Genie system prompt
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "SYSTEM_PROMPT.md"
if SYSTEM_PROMPT_PATH.exists():
    SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
else:
    SYSTEM_PROMPT = (
        "Genie is the assistant for FTU (Trường Đại học Ngoại thương). "
        "Answers FTU facts using only tool results retrieved via babel_query and babel_fetch."
    )

# Tool specs for OpenWebUI function calling
BABEL_TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "babel_query",
            "description": (
                "Search the FTU (Trường Đại học Ngoại thương) knowledge base. "
                "Returns relevant document passages with their doc_id, title, URL, date, and score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string (use user's literal wording, typos included)."
                    },
                    "mode": {
                        "type": "string",
                        "default": "semantic",
                        "enum": ["semantic", "keyword", "hybrid"],
                        "description": "Search mode: semantic, keyword, or hybrid."
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of passage results to return."
                    },
                    "format": {
                        "type": "string",
                        "default": "text",
                        "enum": ["text", "json"],
                        "description": "Output format."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "babel_fetch",
            "description": (
                "Fetch the full document content and metadata for one or more doc_ids in a single batch request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "integer",
                        "description": "A single document ID or list of doc_ids returned by babel_query."
                    },
                    "ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of doc_ids to fetch in batch."
                    },
                    "format": {
                        "type": "string",
                        "default": "text",
                        "enum": ["text", "json"]
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the public web for real-time, external, or non-FTU information via SearXNG. "
                "Use as fallback when FTU knowledge base (babel_query) does not contain the answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query string."
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of search results to return."
                    },
                    "format": {
                        "type": "string",
                        "default": "text",
                        "enum": ["text", "json"]
                    }
                },
                "required": ["query"]
            }
        }
    }
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


async def execute_tool_call(tool_handler: BabelToolHandler, name: str, args_str: str) -> str:
    """Execute a tool call using the local/remote Babel client."""
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
    except Exception as e:
        return f"Error parsing tool arguments: {e}"

    try:
        if name == "babel_query":
            return await asyncio.to_thread(tool_handler.babel_query, **args)
        elif name == "babel_fetch":
            return await asyncio.to_thread(tool_handler.babel_fetch, **args)
        elif name == "web_search":
            return await asyncio.to_thread(tool_handler.web_search, **args)
        else:
            return f"Error: Tool '{name}' not found."
    except Exception as exc:
        return f"Tool execution error: {exc}"


async def run_genie_agent_turn(
    client: httpx.AsyncClient,
    model: str,
    user_messages: list[dict[str, Any]],
    tool_handler: BabelToolHandler,
    max_tool_iterations: int = 5,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """
    Run full Genie Alpha turn with multi-step tool execution loop.
    Returns final text content, latency, token counts, and list of tool calls made.
    """
    headers = {
        "Authorization": f"Bearer {OPENWEBUI_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{OPENWEBUI_ENDPOINT}/api/chat/completions"

    # Assemble conversation messages with system prompt
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(user_messages)

    t0 = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    executed_tools: list[dict[str, Any]] = []

    for _ in range(max_tool_iterations):
        payload = {
            "model": model,
            "messages": messages,
            "tools": BABEL_TOOLS_SPEC,
            "stream": False,
        }

        try:
            r = await client.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code != 200:
                return {
                    "content": "",
                    "latency": time.time() - t0,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "tools_called": executed_tools,
                    "error": f"OpenWebUI HTTP {r.status_code}: {r.text[:200]}",
                }

            data = r.json()
            choice = data["choices"][0]
            msg = choice.get("message", {})
            usage = data.get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                # Terminal answer reached
                return {
                    "content": (msg.get("content") or "").strip(),
                    "latency": time.time() - t0,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "tools_called": executed_tools,
                    "error": None,
                }

            # Append assistant message containing tool calls
            messages.append(msg)

            # Execute tool calls concurrently
            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name", "")
                fn_args = tc.get("function", {}).get("arguments", "{}")
                executed_tools.append({"name": fn_name, "args": fn_args})

                tool_result = await execute_tool_call(tool_handler, fn_name, fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": tool_result,
                })

        except Exception as exc:
            return {
                "content": "",
                "latency": time.time() - t0,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "tools_called": executed_tools,
                "error": str(exc),
            }

    # If max iterations exceeded, return last message content
    return {
        "content": (messages[-1].get("content") or "").strip(),
        "latency": time.time() - t0,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "tools_called": executed_tools,
        "error": "Max tool iterations reached",
    }


async def call_judge_with_retry(
    client: httpx.AsyncClient,
    judge_model: str,
    question_text: str,
    answer_text: str,
    expected_behavior: str,
    grading_notes: str,
    model_response: str,
    max_retries: int = 3,
    timeout: float = 30.0,
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
        "max_tokens": 350,
        "response_format": {"type": "json_object"},
    }

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = await client.post(DEEPINFRA_API_URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                raw_json = r.json()["choices"][0]["message"]["content"]
                parsed = json.loads(raw_json)
                factual = float(parsed.get("factual_accuracy", 0.0))
                grounded = float(parsed.get("groundedness", 0.0))
                complete = float(parsed.get("completeness", 0.0))
                adhere = float(parsed.get("adherence", 0.0))
                final_score = float(parsed.get("final_score", 0.0))
                verdict = parsed.get("verdict", "INCORRECT").upper()
                reason = parsed.get("reason", "")
                return {
                    "factual_accuracy": factual,
                    "groundedness": grounded,
                    "completeness": complete,
                    "adherence": adhere,
                    "final_score": final_score,
                    "verdict": verdict,
                    "reason": reason,
                    "error": None,
                }
            elif r.status_code in (429, 502, 503, 504):
                await asyncio.sleep((attempt + 1) * 2)
                continue
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
        except Exception as e:
            last_err = e
            await asyncio.sleep((attempt + 1) * 2)

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
) -> dict[str, Any]:
    """Process a single benchmark item through Genie tool turn and multi-factor judge evaluation."""
    async with sem:
        # Prepare messages
        if item.get("conversation"):
            messages = item["conversation"]
            q_text = item["conversation"][-1]["content"] if item["conversation"] else item.get("question", "")
        else:
            q_text = item.get("question", "")
            messages = [{"role": "user", "content": q_text}]

        # 1. Run Genie turn with Babel tools
        turn_res = await run_genie_agent_turn(
            client=client,
            model=model,
            user_messages=messages,
            tool_handler=tool_handler,
        )
        model_output = turn_res["content"]

        # 2. Query LLM Judge
        if turn_res["error"] and not model_output:
            judge_res = {
                "factual_accuracy": 0.0,
                "groundedness": 0.0,
                "completeness": 0.0,
                "adherence": 0.0,
                "final_score": 0.0,
                "verdict": "ERROR",
                "reason": f"Inference error: {turn_res['error']}",
                "error": turn_res["error"],
            }
        else:
            judge_res = await call_judge_with_retry(
                client=client,
                judge_model=judge_model,
                question_text=q_text,
                answer_text=item.get("answer", ""),
                expected_behavior=item.get("expected_behavior", "answer"),
                grading_notes=item.get("grading_notes", ""),
                model_response=model_output,
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

    avg_final = sum(r["judge_score"] for r in results) / total
    avg_factual = sum(r.get("factual_accuracy", 0.0) for r in results) / total
    avg_grounded = sum(r.get("groundedness", 0.0) for r in results) / total
    avg_complete = sum(r.get("completeness", 0.0) for r in results) / total
    avg_adhere = sum(r.get("adherence", 0.0) for r in results) / total

    verdict_counts = {}
    for r in results:
        v = r.get("judge_verdict", "UNKNOWN")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    def breakdown_by(field: str):
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            val = r.get(field) or "Unspecified"
            groups.setdefault(val, []).append(r)

        breakdown = {}
        for k, group in groups.items():
            g_total = len(group)
            g_score = sum(x["judge_score"] for x in group)
            breakdown[k] = {
                "count": g_total,
                "avg_final_score": round(g_score / g_total, 3),
                "avg_factual": round(sum(x.get("factual_accuracy", 0.0) for x in group) / g_total, 3),
                "avg_grounded": round(sum(x.get("groundedness", 0.0) for x in group) / g_total, 3),
                "correct": sum(1 for x in group if x["judge_verdict"] == "CORRECT"),
                "partial": sum(1 for x in group if x["judge_verdict"] == "PARTIAL"),
                "incorrect": sum(1 for x in group if x["judge_verdict"] == "INCORRECT"),
            }
        return breakdown

    latencies = [r["model_latency_sec"] for r in results if r.get("model_latency_sec")]
    latencies.sort()
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p50_latency = latencies[int(len(latencies) * 0.50)] if latencies else 0.0
    p90_latency = latencies[int(len(latencies) * 0.90)] if latencies else 0.0

    total_tools_called = sum(len(r.get("tools_called", [])) for r in results)

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


async def main():
    parser = argparse.ArgumentParser(description="Run FTU benchmark against Genie Alpha with Babel Tooling")
    parser.add_argument("--bench-file", default="bench/ftu500.json", help="Path to benchmark JSON file")
    parser.add_argument("--output-jsonl", default="bench/results_genie_alpha.jsonl", help="Output JSONL path")
    parser.add_argument("--output-summary", default="bench/summary_genie_alpha.json", help="Output summary JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Target model in OpenWebUI")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model on DeepInfra")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent requests")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit of questions to benchmark (0 = all)")
    args = parser.parse_args()

    bench_path = Path(args.bench_file)
    if not bench_path.exists():
        print(f"Error: {bench_path} not found.")
        sys.exit(1)

    with open(bench_path, "r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    if args.limit > 0:
        data = data[: args.limit]

    output_jsonl = Path(args.output_jsonl)
    completed_ids = set()
    existing_results: list[dict[str, Any]] = []

    if output_jsonl.exists():
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if "factual_accuracy" in record:
                            completed_ids.add(record["id"])
                            existing_results.append(record)
                    except Exception:
                        pass
        print(f"Resuming benchmark: Found {len(completed_ids)} already completed items in {output_jsonl}")

    remaining_items = [d for d in data if d["id"] not in completed_ids]
    print(f"Total benchmark items: {len(data)} | Remaining: {len(remaining_items)}")

    if not remaining_items and existing_results:
        print("All items already completed. Generating summary...")
        summary = compute_summary_statistics(existing_results)
        with open(args.output_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Summary written to {args.output_summary}")
        return

    sem = asyncio.Semaphore(args.concurrency)
    tool_handler = BabelToolHandler()

    out_file = open(output_jsonl, "a" if completed_ids else "w", encoding="utf-8")
    running_results = list(existing_results)

    async with httpx.AsyncClient(timeout=120.0) as client:
        pbar = tqdm(total=len(data), initial=len(completed_ids), desc=f"Benchmarking {args.model}")

        async def worker(item: dict[str, Any]):
            res = await evaluate_single_item(
                item=item,
                model=args.model,
                judge_model=args.judge_model,
                client=client,
                tool_handler=tool_handler,
                sem=sem,
            )
            out_file.write(json.dumps(res, ensure_ascii=False) + "\n")
            out_file.flush()
            running_results.append(res)
            pbar.update(1)

            cur_scores = [x["judge_score"] for x in running_results]
            cur_avg = sum(cur_scores) / len(cur_scores)
            pbar.set_postfix({"avg_score": f"{cur_avg:.2f}", "tools": len(res.get("tools_called", []))})

        tasks = [worker(item) for item in remaining_items]
        await asyncio.gather(*tasks)

        pbar.close()

    out_file.close()

    print("\nComputing final benchmark summary...")
    summary = compute_summary_statistics(running_results)
    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n================ BENCHMARK COMPLETED ================")
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
