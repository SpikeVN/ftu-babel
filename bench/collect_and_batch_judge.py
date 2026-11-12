#!/usr/bin/env python3
"""
Benchmark answer collector and batch job generator.
Collects answers from both `genie-01-alpha` (with active Babel tool loop)
and `ftu-assistant` for the FTU500 benchmark.
Generates an OpenAI/DeepInfra compatible batch job JSONL file for LLM Judge
(deepseek-ai/DeepSeek-V4-Flash-0731) with float multi-factor scoring.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from oui_tool import Tools as BabelToolHandler

OPENWEBUI_ENDPOINT = os.getenv("OPENWEBUI_ENDPOINT", "").rstrip("/")
OPENWEBUI_KEY = os.getenv("OPENWEBUI_KEY", "")
DEEPINFRA_AUTH_TOKEN = os.getenv("DEEPINFRA_AUTH_TOKEN", "")

DEEPINFRA_API_URL = "https://api.deepinfra.com/v1"
JUDGE_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "SYSTEM_PROMPT.md"
if SYSTEM_PROMPT_PATH.exists():
    GENIE_SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
else:
    GENIE_SYSTEM_PROMPT = (
        "Genie is the assistant for FTU (Trường Đại học Ngoại thương). "
        "Answers FTU facts using only tool results retrieved via babel_query and babel_fetch."
    )

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
    """Execute a tool call using Babel."""
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


async def call_genie_with_babel(
    client: httpx.AsyncClient,
    messages: list[dict[str, Any]],
    tool_handler: BabelToolHandler,
    max_tool_iterations: int = 3,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Call Genie Alpha with active Babel tool execution loop."""
    headers = {
        "Authorization": f"Bearer {OPENWEBUI_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{OPENWEBUI_ENDPOINT}/api/chat/completions"

    full_messages: list[dict[str, Any]] = [{"role": "system", "content": GENIE_SYSTEM_PROMPT}]
    full_messages.extend(messages)

    t0 = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    executed_tools: list[dict[str, Any]] = []

    for _ in range(max_tool_iterations):
        payload = {
            "model": "genie-01-alpha",
            "messages": full_messages,
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
                    "tools": executed_tools,
                    "error": f"HTTP {r.status_code}: {r.text[:200]}",
                }

            data = r.json()
            choice = data["choices"][0]
            msg = choice.get("message", {})
            usage = data.get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                return {
                    "content": (msg.get("content") or "").strip(),
                    "latency": time.time() - t0,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "tools": executed_tools,
                    "error": None,
                }

            full_messages.append(msg)
            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name", "")
                fn_args = tc.get("function", {}).get("arguments", "{}")
                executed_tools.append({"name": fn_name, "args": fn_args})

                res_str = await execute_tool_call(tool_handler, fn_name, fn_args)
                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": res_str,
                })

        except Exception as exc:
            return {
                "content": "",
                "latency": time.time() - t0,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "tools": executed_tools,
                "error": str(exc),
            }

    return {
        "content": (full_messages[-1].get("content") or "").strip(),
        "latency": time.time() - t0,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "tools": executed_tools,
        "error": "Max tool iterations reached",
    }


async def call_ftu_assistant(
    client: httpx.AsyncClient,
    messages: list[dict[str, Any]],
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Call FTU Assistant model on OpenWebUI."""
    headers = {
        "Authorization": f"Bearer {OPENWEBUI_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{OPENWEBUI_ENDPOINT}/api/chat/completions"

    t0 = time.time()
    payload = {
        "model": "ftu-assistant",
        "messages": messages,
        "stream": False,
    }

    try:
        r = await client.post(url, headers=headers, json=payload, timeout=timeout)
        latency = time.time() - t0
        if r.status_code == 200:
            data = r.json()
            choice = data["choices"][0]
            msg = choice.get("message", {})
            usage = data.get("usage", {})
            return {
                "content": (msg.get("content") or "").strip(),
                "latency": latency,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "error": None,
            }
        else:
            return {
                "content": "",
                "latency": latency,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": f"HTTP {r.status_code}: {r.text[:200]}",
            }
    except Exception as exc:
        return {
            "content": "",
            "latency": time.time() - t0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": str(exc),
        }


async def process_item(
    item: dict[str, Any],
    client: httpx.AsyncClient,
    tool_handler: BabelToolHandler,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Collect responses from both models for a single question."""
    async with sem:
        if item.get("conversation"):
            messages = item["conversation"]
            q_text = item["conversation"][-1]["content"] if item["conversation"] else item.get("question", "")
        else:
            q_text = item.get("question", "")
            messages = [{"role": "user", "content": q_text}]

        # Run both model queries concurrently
        genie_task = call_genie_with_babel(client, messages, tool_handler)
        ftu_task = call_ftu_assistant(client, messages)

        genie_res, ftu_res = await asyncio.gather(genie_task, ftu_task)

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
            "sources": item.get("sources", []),
            "genie_output": genie_res["content"],
            "genie_latency_sec": round(genie_res["latency"], 3),
            "genie_tools": genie_res.get("tools", []),
            "genie_prompt_tokens": genie_res["prompt_tokens"],
            "genie_completion_tokens": genie_res["completion_tokens"],
            "genie_error": genie_res["error"],
            "ftu_assistant_output": ftu_res["content"],
            "ftu_assistant_latency_sec": round(ftu_res["latency"], 3),
            "ftu_assistant_prompt_tokens": ftu_res["prompt_tokens"],
            "ftu_assistant_completion_tokens": ftu_res["completion_tokens"],
            "ftu_assistant_error": ftu_res["error"],
        }


def make_judge_request(
    custom_id: str,
    question: str,
    ground_truth: str,
    expected_behavior: str,
    grading_notes: str,
    model_response: str,
) -> dict[str, Any]:
    """Create an OpenAI/DeepInfra Batch API compatible request object."""
    user_prompt = f"""[CÂU HỎI]: {question}
[ĐÁP ÁN THAM CHIẾU]: {ground_truth}
[HÀNH VI KỲ VỌNG]: {expected_behavior}
[HƯỚNG DẪN CHẤM]: {grading_notes}
[CÂU TRẢ LỜI CỦA MODEL]: {model_response}"""

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 350,
            "response_format": {"type": "json_object"},
        },
    }


def generate_batch_file(collected_file: Path, batch_output_file: Path) -> int:
    """Generate the standard OpenAI/DeepInfra Batch API input JSONL file from collected responses."""
    count = 0
    with open(collected_file, "r", encoding="utf-8") as fin, open(batch_output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec["id"]
            q = rec["question"]
            gt = rec["answer"]
            eb = rec.get("expected_behavior", "answer")
            gn = rec.get("grading_notes", "")

            # 1. Judge request for Genie Alpha
            req_genie = make_judge_request(
                custom_id=f"genie-01-alpha__{qid}",
                question=q,
                ground_truth=gt,
                expected_behavior=eb,
                grading_notes=gn,
                model_response=rec.get("genie_output", ""),
            )
            fout.write(json.dumps(req_genie, ensure_ascii=False) + "\n")
            count += 1

            # 2. Judge request for FTU Assistant
            req_ftu = make_judge_request(
                custom_id=f"ftu-assistant__{qid}",
                question=q,
                ground_truth=gt,
                expected_behavior=eb,
                grading_notes=gn,
                model_response=rec.get("ftu_assistant_output", ""),
            )
            fout.write(json.dumps(req_ftu, ensure_ascii=False) + "\n")
            count += 1

    return count


async def upload_and_create_deepinfra_batch(batch_file: Path) -> dict[str, Any]:
    """Upload batch file and submit batch job to DeepInfra /v1/batches."""
    headers = {"Authorization": f"Bearer {DEEPINFRA_AUTH_TOKEN}"}

    async with httpx.AsyncClient(timeout=120.0) as client:
        print(f"Uploading {batch_file} to DeepInfra /v1/files...")
        with open(batch_file, "rb") as f:
            files = {"file": (batch_file.name, f, "application/jsonl")}
            data = {"purpose": "batch"}
            r_upload = await client.post(f"{DEEPINFRA_API_URL}/files", headers=headers, files=files, data=data)

        if r_upload.status_code not in (200, 201):
            raise RuntimeError(f"File upload failed ({r_upload.status_code}): {r_upload.text}")

        file_info = r_upload.json()
        file_id = file_info["id"]
        print(f"Uploaded file successfully. File ID: {file_id}")

        print("Creating batch job on DeepInfra /v1/batches...")
        batch_payload = {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": {
                "benchmark": "FTU500",
                "models": "genie-01-alpha,ftu-assistant",
                "judge": JUDGE_MODEL,
            },
        }

        r_batch = await client.post(f"{DEEPINFRA_API_URL}/batches", headers=headers, json=batch_payload)
        if r_batch.status_code not in (200, 201):
            raise RuntimeError(f"Batch creation failed ({r_batch.status_code}): {r_batch.text}")

        batch_info = r_batch.json()
        print(f"Batch job created successfully! Batch ID: {batch_info.get('id')}")
        return batch_info


async def main():
    parser = argparse.ArgumentParser(description="Collect answers and create DeepInfra/OpenAI batch job")
    parser.add_argument("--bench-file", default="bench/ftu500.json", help="Path to benchmark JSON")
    parser.add_argument("--output-collected", default="bench/collected_answers.jsonl", help="Collected answers JSONL")
    parser.add_argument("--output-batch", default="bench/judge_batch_input.jsonl", help="Batch requests JSONL")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent requests")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for testing")
    parser.add_argument("--submit", action="store_true", help="Submit batch to DeepInfra /v1/batches")
    args = parser.parse_args()

    bench_path = Path(args.bench_file)
    if not bench_path.exists():
        print(f"Error: {bench_path} not found.")
        sys.exit(1)

    with open(bench_path, "r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    if args.limit > 0:
        data = data[: args.limit]

    collected_path = Path(args.output_collected)
    completed_ids = set()
    if collected_path.exists():
        with open(collected_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        completed_ids.add(record["id"])
                    except Exception:
                        pass
        print(f"Resuming: Found {len(completed_ids)} already collected items.")

    remaining_items = [d for d in data if d["id"] not in completed_ids]
    print(f"Total benchmark items: {len(data)} | Remaining to collect: {len(remaining_items)}")

    if remaining_items:
        sem = asyncio.Semaphore(args.concurrency)
        tool_handler = BabelToolHandler()
        out_file = open(collected_path, "a" if completed_ids else "w", encoding="utf-8")

        async with httpx.AsyncClient(timeout=120.0) as client:
            pbar = tqdm(total=len(data), initial=len(completed_ids), desc="Collecting Answers")

            async def worker(item: dict[str, Any]):
                res = await process_item(item, client, tool_handler, sem)
                out_file.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_file.flush()
                pbar.update(1)

            tasks = [worker(item) for item in remaining_items]
            await asyncio.gather(*tasks)
            pbar.close()

        out_file.close()

    print(f"\nGenerating OpenAI/DeepInfra compatible batch file: {args.output_batch}...")
    batch_path = Path(args.output_batch)
    total_reqs = generate_batch_file(collected_path, batch_path)
    print(f"Successfully generated {total_reqs} batch requests in {batch_path}")

    if args.submit:
        batch_res = await upload_and_create_deepinfra_batch(batch_path)
        batch_id = batch_res.get("id")
        print(f"\nBatch ID: {batch_id}")
        print(f"Check status with:")
        print(f"  curl -s -H 'Authorization: Bearer $DEEPINFRA_AUTH_TOKEN' https://api.deepinfra.com/v1/batches/{batch_id}")


if __name__ == "__main__":
    asyncio.run(main())
