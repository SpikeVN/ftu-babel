#!/usr/bin/env python3
"""
Benchmark answer collector and DeepInfra batch job generator.
Collects answers from `genie-01-alpha` (with active Babel tool loop) and/or
`ftu-assistant` for the FTU500 benchmark, and generates an OpenAI/DeepInfra
Batch API JSONL file for LLM Judge (deepseek-ai/DeepSeek-V4-Flash-0731).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import httpx
from dotenv import load_dotenv
from tqdm import tqdm as sync_tqdm
from tqdm.asyncio import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from bench.run_benchmark import (  # noqa: E402
    ANSI_BLUE,
    BABEL_TOOLS_SPEC,
    DEEPINFRA_AUTH_TOKEN,
    DEFAULT_JUDGE_MODEL as JUDGE_MODEL,
    DEFAULT_TOOL_IDS,
    JUDGE_SYSTEM_PROMPT,
    OPENWEBUI_ENDPOINT,
    OPENWEBUI_KEY,
    SYSTEM_PROMPT as GENIE_SYSTEM_PROMPT,
    compact_preview,
    execute_tool_call,
    extract_user_messages,
    format_colored_log,
    format_question_for_judge,
    post_json_with_retry,
    run_genie_agent_turn,
    sanitize_model_response,
)
from oui_tool import Tools as BabelToolHandler  # noqa: E402

DEEPINFRA_API_URL = "https://api.deepinfra.com/v1"
SUPPORTED_MODELS = ("genie-01-alpha", "ftu-assistant")


async def call_genie_with_babel(
    client: httpx.AsyncClient,
    messages: list[dict[str, Any]],
    tool_handler: BabelToolHandler,
    max_tool_iterations: int = 8,
    timeout: float = 90.0,
    use_local_prompt: bool = False,
    tool_ids: list[str] | None = None,
    item_id: str | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Call Genie Alpha with active parallel Babel tool execution loop."""
    res = await run_genie_agent_turn(
        client=client,
        model="genie-01-alpha",
        user_messages=messages,
        tool_handler=tool_handler,
        max_tool_iterations=max_tool_iterations,
        timeout=timeout,
        use_local_prompt=use_local_prompt,
        tool_ids=tool_ids,
        item_id=item_id,
        log_fn=log_fn,
    )
    return {
        "content": res["content"],
        "latency": res["latency"],
        "prompt_tokens": res["prompt_tokens"],
        "completion_tokens": res["completion_tokens"],
        "tools": res["tools_called"],
        "error": res["error"],
    }


async def call_ftu_assistant(
    client: httpx.AsyncClient,
    messages: list[dict[str, Any]],
    timeout: float = 90.0,
    max_retries: int = 3,
    item_id: str | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Call FTU Assistant model on OpenWebUI with transient retry handling."""
    prefix = f"[ftu-assistant][{item_id}]" if item_id else "[ftu-assistant]"

    def _log(text: str) -> None:
        if log_fn is not None:
            log_fn(format_colored_log(f"{prefix} {text}", ANSI_BLUE))

    headers = {
        "Authorization": f"Bearer {OPENWEBUI_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{OPENWEBUI_ENDPOINT}/api/chat/completions"

    last_user_msg = (messages[-1].get("content") or "") if messages else ""
    _log(f"Q: {compact_preview(last_user_msg, 160)}")

    t0 = time.time()
    payload = {
        "model": "ftu-assistant",
        "messages": messages,
        "stream": False,
    }

    try:
        r = await post_json_with_retry(
            client, url, headers, payload, timeout=timeout, max_retries=max_retries
        )
        latency = time.time() - t0
        if r.status_code == 200:
            data = r.json()
            choice = data["choices"][0]
            msg = choice.get("message", {})
            usage = data.get("usage") or {}
            content = sanitize_model_response(msg.get("content") or "")
            prompt_toks = usage.get("prompt_tokens", 0)
            completion_toks = usage.get("completion_tokens", 0)
            _log(f"A ({latency:.1f}s, {prompt_toks}->{completion_toks} tok): {compact_preview(content, 240)}")
            return {
                "content": content,
                "latency": latency,
                "prompt_tokens": prompt_toks,
                "completion_tokens": completion_toks,
                "error": None,
            }
        err_msg = f"HTTP {r.status_code}: {r.text[:200]}"
        _log(f"ERR ({latency:.1f}s): {compact_preview(err_msg, 200)}")
        return {
            "content": "",
            "latency": latency,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": err_msg,
        }
    except Exception as exc:
        latency = time.time() - t0
        _log(f"ERR ({latency:.1f}s): {compact_preview(str(exc), 200)}")
        return {
            "content": "",
            "latency": latency,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": str(exc),
        }


async def process_item(
    item: dict[str, Any],
    client: httpx.AsyncClient,
    tool_handler: BabelToolHandler,
    sem: asyncio.Semaphore,
    *,
    selected_models: list[str] | None = None,
    use_local_prompt: bool = False,
    tool_ids: list[str] | None = None,
    max_tool_iterations: int = 8,
    existing_record: dict[str, Any] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Collect responses from the selected models for a single benchmark question."""
    models = selected_models or list(SUPPORTED_MODELS)

    async with sem:
        messages, q_text = extract_user_messages(item)

        genie_res: dict[str, Any] | None = None
        ftu_res: dict[str, Any] | None = None

        tasks: list[tuple[str, Any]] = []
        if "genie-01-alpha" in models:
            tasks.append((
                "genie-01-alpha",
                call_genie_with_babel(
                    client,
                    messages,
                    tool_handler,
                    max_tool_iterations=max_tool_iterations,
                    use_local_prompt=use_local_prompt,
                    tool_ids=tool_ids,
                    item_id=item["id"],
                    log_fn=log_fn,
                ),
            ))
        if "ftu-assistant" in models:
            tasks.append((
                "ftu-assistant",
                call_ftu_assistant(
                    client,
                    messages,
                    item_id=item["id"],
                    log_fn=log_fn,
                ),
            ))

        results = await asyncio.gather(*(coro for _, coro in tasks))
        for name, res in zip((n for n, _ in tasks), results):
            if name == "genie-01-alpha":
                genie_res = res
            elif name == "ftu-assistant":
                ftu_res = res

        prev = existing_record or {}
        record: dict[str, Any] = {
            "id": item["id"],
            "type": item.get("type"),
            "subtype": item.get("subtype"),
            "difficulty": item.get("difficulty"),
            "category": item.get("category"),
            "question": q_text,
            "question_for_judge": format_question_for_judge(item),
            "conversation": item.get("conversation"),
            "answer": item.get("answer", ""),
            "expected_behavior": item.get("expected_behavior", "answer"),
            "grading_notes": item.get("grading_notes", ""),
            "sources": item.get("sources", []),
        }

        if genie_res is not None:
            record.update({
                "genie_output": genie_res["content"],
                "genie_latency_sec": round(genie_res["latency"], 3),
                "genie_tools": genie_res.get("tools", []),
                "genie_prompt_tokens": genie_res["prompt_tokens"],
                "genie_completion_tokens": genie_res["completion_tokens"],
                "genie_error": genie_res["error"],
            })
        elif "genie_output" in prev:
            for k in (
                "genie_output",
                "genie_latency_sec",
                "genie_tools",
                "genie_prompt_tokens",
                "genie_completion_tokens",
                "genie_error",
            ):
                record[k] = prev.get(k)

        if ftu_res is not None:
            record.update({
                "ftu_assistant_output": ftu_res["content"],
                "ftu_assistant_latency_sec": round(ftu_res["latency"], 3),
                "ftu_assistant_prompt_tokens": ftu_res["prompt_tokens"],
                "ftu_assistant_completion_tokens": ftu_res["completion_tokens"],
                "ftu_assistant_error": ftu_res["error"],
            })
        elif "ftu_assistant_output" in prev:
            for k in (
                "ftu_assistant_output",
                "ftu_assistant_latency_sec",
                "ftu_assistant_prompt_tokens",
                "ftu_assistant_completion_tokens",
                "ftu_assistant_error",
            ):
                record[k] = prev.get(k)

        return record


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
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        },
    }


def _model_response_fields(model_name: str) -> tuple[str, str]:
    if model_name == "genie-01-alpha":
        return "genie_output", "genie_error"
    if model_name == "ftu-assistant":
        return "ftu_assistant_output", "ftu_assistant_error"
    raise ValueError(f"Unsupported model: {model_name}")


def _item_models_succeeded(record: dict[str, Any], selected_models: list[str]) -> bool:
    for model_name in selected_models:
        out_key, err_key = _model_response_fields(model_name)
        if record.get(err_key) or not (record.get(out_key) or "").strip():
            return False
    return True


def generate_batch_file(
    collected_file: Path,
    batch_output_file: Path,
    selected_models: list[str] | None = None,
    bench_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[int, int]:
    """
    Generate the DeepInfra/OpenAI Batch API input JSONL from collected responses.
    Deduplicates collected records by `id` (latest wins) and skips errored/empty responses.
    Returns (emitted_count, skipped_count).
    """
    models = selected_models or list(SUPPORTED_MODELS)
    records_by_id: dict[str, dict[str, Any]] = {}

    with open(collected_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "id" in rec:
                    records_by_id[rec["id"]] = rec
            except Exception:
                pass

    emitted = 0
    skipped = 0
    batch_output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(batch_output_file, "w", encoding="utf-8") as fout:
        for qid, rec in sync_tqdm(
            records_by_id.items(),
            total=len(records_by_id),
            desc="Generating Batch JSONL",
            unit="item",
        ):
            if rec.get("question_for_judge"):
                judge_q = rec["question_for_judge"]
            elif rec.get("conversation"):
                judge_q = format_question_for_judge(rec)
            elif bench_by_id and qid in bench_by_id:
                judge_q = format_question_for_judge(bench_by_id[qid])
            else:
                judge_q = rec.get("question", "")

            gt = rec.get("answer", "")
            eb = rec.get("expected_behavior", "answer")
            gn = rec.get("grading_notes", "")

            for model_name in models:
                out_key, err_key = _model_response_fields(model_name)
                output_text = (rec.get(out_key) or "").strip()
                err_val = rec.get(err_key)
                if err_val or not output_text:
                    skipped += 1
                    continue

                req = make_judge_request(
                    custom_id=f"{model_name}__{qid}",
                    question=judge_q,
                    ground_truth=gt,
                    expected_behavior=eb,
                    grading_notes=gn,
                    model_response=output_text,
                )
                fout.write(json.dumps(req, ensure_ascii=False) + "\n")
                emitted += 1

    return emitted, skipped


async def upload_and_create_deepinfra_batch(
    batch_file: Path,
    selected_models: list[str] | None = None,
) -> dict[str, Any]:
    """Upload batch file and submit batch job to DeepInfra /v1/batches."""
    headers = {"Authorization": f"Bearer {DEEPINFRA_AUTH_TOKEN}"}
    models_str = ",".join(selected_models or list(SUPPORTED_MODELS))

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
                "models": models_str,
                "judge": JUDGE_MODEL,
            },
        }

        r_batch = await client.post(f"{DEEPINFRA_API_URL}/batches", headers=headers, json=batch_payload)
        if r_batch.status_code not in (200, 201):
            raise RuntimeError(f"Batch creation failed ({r_batch.status_code}): {r_batch.text}")

        batch_info = r_batch.json()
        print(f"Batch job created successfully! Batch ID: {batch_info.get('id')}")
        return batch_info


async def main() -> None:
    parser = argparse.ArgumentParser(description="Collect answers and create DeepInfra/OpenAI batch job")
    parser.add_argument("--bench-file", default="bench/ftu500.json", help="Path to benchmark JSON")
    parser.add_argument("--output-collected", default="bench/collected_answers.jsonl", help="Collected answers JSONL")
    parser.add_argument("--output-batch", default="bench/judge_batch_input.jsonl", help="Batch requests JSONL")
    parser.add_argument(
        "--models",
        default=",".join(SUPPORTED_MODELS),
        help="Comma-separated models to collect (genie-01-alpha,ftu-assistant)",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent requests")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for testing")
    parser.add_argument("--ids", default="", help="Comma-separated question IDs to run (e.g. Q001,MT-001)")
    parser.add_argument(
        "--use-local-prompt",
        action="store_true",
        help="Send local SYSTEM_PROMPT.md and BABEL_TOOLS_SPEC instead of OpenWebUI server-side prompt & tool_ids",
    )
    parser.add_argument(
        "--tool-ids",
        default=",".join(DEFAULT_TOOL_IDS),
        help="Comma-separated OpenWebUI tool IDs for genie-01-alpha (used when --use-local-prompt is off)",
    )
    parser.add_argument("--max-tool-iterations", type=int, default=8, help="Max tool-calling rounds for genie-01-alpha")
    parser.add_argument("--quiet", action="store_true", help="Disable live colored chat and tool logs")
    parser.add_argument("--submit", action="store_true", help="Submit batch to DeepInfra /v1/batches")
    args = parser.parse_args()

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    invalid_models = [m for m in selected_models if m not in SUPPORTED_MODELS]
    if not selected_models or invalid_models:
        print(f"Error: Unsupported --models {invalid_models or selected_models}. Choose from {list(SUPPORTED_MODELS)}.")
        sys.exit(1)

    if not OPENWEBUI_ENDPOINT or not OPENWEBUI_KEY:
        print("Error: OPENWEBUI_ENDPOINT and OPENWEBUI_KEY must be set in .env or environment.")
        sys.exit(1)
    if args.submit and not DEEPINFRA_AUTH_TOKEN:
        print("Error: DEEPINFRA_AUTH_TOKEN must be set when using --submit.")
        sys.exit(1)

    bench_path = Path(args.bench_file)
    if not bench_path.exists():
        print(f"Error: {bench_path} not found.")
        sys.exit(1)

    with open(bench_path, "r", encoding="utf-8") as f:
        full_data: list[dict[str, Any]] = json.load(f)

    bench_by_id = {d["id"]: d for d in full_data}
    data = list(full_data)
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

    collected_path = Path(args.output_collected)
    collected_path.parent.mkdir(parents=True, exist_ok=True)

    records_by_id: dict[str, dict[str, Any]] = {}
    if collected_path.exists():
        with open(collected_path, "r", encoding="utf-8") as f:
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

    completed_ids = {
        qid for qid, rec in records_by_id.items()
        if qid in target_ids and _item_models_succeeded(rec, selected_models)
    }
    if records_by_id:
        print(
            f"Resuming: Found {len(completed_ids)} already completed items for {selected_models} "
            f"(out of {len(records_by_id)} records)."
        )

    remaining_items = [d for d in data if d["id"] not in completed_ids]
    print(f"Models: {selected_models} | Total items: {len(data)} | Remaining to collect: {len(remaining_items)}")

    if remaining_items:
        sem = asyncio.Semaphore(args.concurrency)
        tool_handler = BabelToolHandler()
        file_mode = "a" if collected_path.exists() and records_by_id else "w"

        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(collected_path, file_mode, encoding="utf-8") as out_file:
                pbar = tqdm(
                    total=len(data),
                    initial=len(completed_ids),
                    desc="Collecting Answers",
                    dynamic_ncols=True,
                )
                log_fn = None if args.quiet else (lambda msg: pbar.write(msg, file=sys.stderr))

                async def worker(item: dict[str, Any]) -> None:
                    res = await process_item(
                        item,
                        client,
                        tool_handler,
                        sem,
                        selected_models=selected_models,
                        use_local_prompt=args.use_local_prompt,
                        tool_ids=tool_ids,
                        max_tool_iterations=args.max_tool_iterations,
                        existing_record=records_by_id.get(item["id"]),
                        log_fn=log_fn,
                    )
                    out_file.write(json.dumps(res, ensure_ascii=False) + "\n")
                    out_file.flush()
                    records_by_id[res["id"]] = res
                    pbar.update(1)

                    target_recs = [r for qid, r in records_by_id.items() if qid in target_ids]
                    ok_count = sum(1 for r in target_recs if _item_models_succeeded(r, selected_models))
                    err_count = len(target_recs) - ok_count
                    postfix: dict[str, Any] = {"ok": ok_count, "errs": err_count}
                    if "genie-01-alpha" in selected_models:
                        postfix["tools"] = len(res.get("genie_tools") or [])
                    pbar.set_postfix(postfix)

                await asyncio.gather(*(worker(item) for item in remaining_items))
                pbar.close()

    print(f"\nGenerating OpenAI/DeepInfra compatible batch file: {args.output_batch}...")
    batch_path = Path(args.output_batch)
    total_reqs, skipped_reqs = generate_batch_file(
        collected_path,
        batch_path,
        selected_models=selected_models,
        bench_by_id=bench_by_id,
    )
    print(f"Successfully generated {total_reqs} batch requests in {batch_path} (skipped {skipped_reqs} errored/empty).")

    if args.submit:
        if total_reqs == 0:
            print("Error: Batch file contains 0 requests; skipping upload.")
            sys.exit(1)
        batch_res = await upload_and_create_deepinfra_batch(batch_path, selected_models=selected_models)
        batch_id = batch_res.get("id")
        print(f"\nBatch ID: {batch_id}")
        print("Download & score when complete with:")
        print(f"  .venv/bin/python bench/score_batch_results.py --batch-id {batch_id} --collected {collected_path}")


if __name__ == "__main__":
    asyncio.run(main())
