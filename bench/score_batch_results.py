#!/usr/bin/env python3
"""
Download (optional) and score DeepInfra Batch API judge outputs produced by
`bench/collect_and_batch_judge.py`, generating per-model JSONL files and a
combined multi-factor benchmark summary JSON + comparison table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from bench.run_benchmark import (  # noqa: E402
    DEEPINFRA_AUTH_TOKEN,
    compute_summary_statistics,
    score_and_verdict,
)

DEEPINFRA_API_URL = "https://api.deepinfra.com/v1"


def fetch_batch_output_if_ready(batch_id: str, batch_output_path: Path) -> bool:
    """
    Check DeepInfra batch status for `batch_id`.
    If completed, download `output_file_id` to `batch_output_path` and return True.
    If still pending/in-progress, print status and return False.
    If terminal failure, raise RuntimeError.
    """
    if not DEEPINFRA_AUTH_TOKEN:
        raise RuntimeError("DEEPINFRA_AUTH_TOKEN must be set in .env or environment to query DeepInfra batches.")

    headers = {"Authorization": f"Bearer {DEEPINFRA_AUTH_TOKEN}"}
    with httpx.Client(timeout=120.0) as client:
        r = client.get(f"{DEEPINFRA_API_URL}/batches/{batch_id}", headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to fetch batch {batch_id} (HTTP {r.status_code}): {r.text[:300]}")
        batch_info = r.json()
        status = (batch_info.get("status") or "unknown").lower()
        req_counts = batch_info.get("request_counts") or {}
        print(f"Batch {batch_id} status: {status} | request_counts={req_counts}")
        total_cnt = int(req_counts.get("total") or 0)
        completed_cnt = int(req_counts.get("completed") or 0)
        failed_cnt = int(req_counts.get("failed") or 0)
        if total_cnt > 0:
            pct = (completed_cnt / total_cnt) * 100.0
            print(f"Batch progress: {completed_cnt}/{total_cnt} ({pct:.1f}%) | failed: {failed_cnt}")

        if status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} ended with status '{status}': {batch_info.get('errors')}")

        if status != "completed":
            print("Batch is not completed yet. Re-run this command later to download and score results.")
            return False

        output_file_id = batch_info.get("output_file_id")
        if not output_file_id:
            error_file_id = batch_info.get("error_file_id")
            raise RuntimeError(f"Batch {batch_id} completed without output_file_id (error_file_id={error_file_id}).")

        print(f"Downloading batch output file {output_file_id} -> {batch_output_path}...")
        batch_output_path.parent.mkdir(parents=True, exist_ok=True)
        total_bytes = 0
        with client.stream("GET", f"{DEEPINFRA_API_URL}/files/{output_file_id}/content", headers=headers) as r_file:
            if r_file.status_code != 200:
                body_err = r_file.read().decode("utf-8", errors="replace")[:300]
                raise RuntimeError(
                    f"Failed to download output file {output_file_id} (HTTP {r_file.status_code}): {body_err}"
                )
            content_len = r_file.headers.get("Content-Length")
            total_len = int(content_len) if content_len and content_len.isdigit() else None
            with (
                open(batch_output_path, "wb") as fout,
                tqdm(
                    total=total_len,
                    desc="Downloading Batch Output",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                for chunk in r_file.iter_bytes():
                    if chunk:
                        fout.write(chunk)
                        total_bytes += len(chunk)
                        pbar.update(len(chunk))
        print(f"Saved {total_bytes} bytes to {batch_output_path}.")
        return True


def load_collected_by_id(collected_path: Path) -> dict[str, dict[str, Any]]:
    """Load collected answers JSONL and deduplicate by `id` (latest record wins)."""
    records: dict[str, dict[str, Any]] = {}
    if not collected_path.exists():
        return records

    with open(collected_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "id" in rec:
                    records[rec["id"]] = rec
            except Exception:
                pass
    return records


def _extract_model_fields(collected_rec: dict[str, Any], model_name: str) -> dict[str, Any]:
    if model_name == "genie-01-alpha":
        return {
            "model_response": collected_rec.get("genie_output", ""),
            "tools_called": collected_rec.get("genie_tools", []),
            "model_latency_sec": collected_rec.get("genie_latency_sec", 0.0),
            "prompt_tokens": collected_rec.get("genie_prompt_tokens", 0),
            "completion_tokens": collected_rec.get("genie_completion_tokens", 0),
            "collection_error": collected_rec.get("genie_error"),
        }
    if model_name == "ftu-assistant":
        return {
            "model_response": collected_rec.get("ftu_assistant_output", ""),
            "tools_called": [],
            "model_latency_sec": collected_rec.get("ftu_assistant_latency_sec", 0.0),
            "prompt_tokens": collected_rec.get("ftu_assistant_prompt_tokens", 0),
            "completion_tokens": collected_rec.get("ftu_assistant_completion_tokens", 0),
            "collection_error": collected_rec.get("ftu_assistant_error"),
        }
    return {
        "model_response": "",
        "tools_called": [],
        "model_latency_sec": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "collection_error": None,
    }


def _parse_batch_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Parse one OpenAI/DeepInfra batch output JSON line into standardized judge fields."""
    batch_err = entry.get("error")
    if batch_err is not None:
        return {
            "factual_accuracy": 0.0,
            "groundedness": 0.0,
            "completeness": 0.0,
            "adherence": 0.0,
            "final_score": 0.0,
            "verdict": "ERROR",
            "reason": f"Batch item error: {batch_err}",
            "error": str(batch_err),
        }

    resp = entry.get("response") or {}
    status_code = resp.get("status_code")
    if status_code is not None and int(status_code) != 200:
        body_preview = str(resp.get("body") or "")[:200]
        return {
            "factual_accuracy": 0.0,
            "groundedness": 0.0,
            "completeness": 0.0,
            "adherence": 0.0,
            "final_score": 0.0,
            "verdict": "ERROR",
            "reason": f"Batch response HTTP {status_code}: {body_preview}",
            "error": f"HTTP {status_code}: {body_preview}",
        }

    try:
        body = resp.get("body") or {}
        choices = body.get("choices") or []
        raw_content = (choices[0].get("message") or {}).get("content") or ""
        return score_and_verdict(raw_content)
    except Exception as exc:
        return {
            "factual_accuracy": 0.0,
            "groundedness": 0.0,
            "completeness": 0.0,
            "adherence": 0.0,
            "final_score": 0.0,
            "verdict": "ERROR",
            "reason": f"Batch parse error: {exc}",
            "error": str(exc),
        }


def score_batch_file(
    batch_output_path: Path,
    collected_path: Path,
    output_dir: Path,
    output_summary_path: Path,
) -> dict[str, Any]:
    """Join batch judge outputs with collected records, write per-model files, and return combined summary."""
    collected_by_id = load_collected_by_id(collected_path)
    per_model_records: dict[str, dict[str, dict[str, Any]]] = {}

    with open(batch_output_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for raw in f if raw.strip())

    scored_count = 0
    err_count = 0
    with (
        open(batch_output_path, "r", encoding="utf-8") as f,
        tqdm(total=total_lines, desc="Scoring Batch Output", unit="req") as pbar,
    ):
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            pbar.update(1)
            try:
                entry = json.loads(line)
            except Exception as exc:
                err_count += 1
                pbar.write(f"Warning: Skipping malformed JSON at line {line_num}: {exc}")
                pbar.set_postfix({"scored": scored_count, "errs": err_count})
                continue

            custom_id = str(entry.get("custom_id") or "")
            if "__" not in custom_id:
                err_count += 1
                pbar.write(f"Warning: Skipping entry with invalid custom_id {custom_id!r} at line {line_num}")
                pbar.set_postfix({"scored": scored_count, "errs": err_count})
                continue

            model_name, qid = custom_id.split("__", 1)
            judge_res = _parse_batch_entry(entry)
            base_rec = collected_by_id.get(qid, {})
            model_fields = _extract_model_fields(base_rec, model_name)

            scored_rec = {
                "id": qid,
                "type": base_rec.get("type"),
                "subtype": base_rec.get("subtype"),
                "difficulty": base_rec.get("difficulty"),
                "category": base_rec.get("category"),
                "question": base_rec.get("question", ""),
                "answer": base_rec.get("answer", ""),
                "expected_behavior": base_rec.get("expected_behavior", "answer"),
                "grading_notes": base_rec.get("grading_notes", ""),
                "model_response": model_fields["model_response"],
                "tools_called": model_fields["tools_called"],
                "model_latency_sec": model_fields["model_latency_sec"],
                "prompt_tokens": model_fields["prompt_tokens"],
                "completion_tokens": model_fields["completion_tokens"],
                "factual_accuracy": judge_res["factual_accuracy"],
                "groundedness": judge_res["groundedness"],
                "completeness": judge_res["completeness"],
                "adherence": judge_res["adherence"],
                "judge_score": judge_res["final_score"],
                "judge_verdict": judge_res["verdict"],
                "judge_reason": judge_res["reason"],
                "error": model_fields["collection_error"] or judge_res.get("error"),
            }
            per_model_records.setdefault(model_name, {})[qid] = scored_rec
            scored_count += 1
            if judge_res["verdict"] == "ERROR":
                err_count += 1
            pbar.set_postfix({"scored": scored_count, "errs": err_count})

    output_dir.mkdir(parents=True, exist_ok=True)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)

    combined_summary: dict[str, Any] = {
        "batch_output_file": str(batch_output_path),
        "collected_file": str(collected_path),
        "models": {},
    }

    for model_name, records_map in per_model_records.items():
        records_list = list(records_map.values())
        safe_name = model_name.replace("/", "_")
        model_jsonl = output_dir / f"results_{safe_name}_batch.jsonl"
        with open(model_jsonl, "w", encoding="utf-8") as fout:
            for rec in records_list:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        summary = compute_summary_statistics(records_list)
        combined_summary["models"][model_name] = summary

        model_summary_path = output_dir / f"summary_{safe_name}_batch.json"
        with open(model_summary_path, "w", encoding="utf-8") as fout:
            json.dump(summary, fout, ensure_ascii=False, indent=2)

    with open(output_summary_path, "w", encoding="utf-8") as fout:
        json.dump(combined_summary, fout, ensure_ascii=False, indent=2)

    return combined_summary


def print_comparison_table(combined_summary: dict[str, Any]) -> None:
    """Print a clean side-by-side summary table across all scored models."""
    models = combined_summary.get("models") or {}
    if not models:
        print("No scored model results found in batch output.")
        return

    print("\n" + "=" * 96)
    print("                          FTU500 BATCH EVALUATION SUMMARY")
    print("=" * 96)
    header = (
        f"{'Model':<20} | {'N':>4} | {'Score':>6} | {'Fact':>6} | {'Ground':>6} | "
        f"{'Compl':>6} | {'Adher':>6} | {'COR/PAR/INC/ERR':<17} | {'Lat(s)':>6}"
    )
    print(header)
    print("-" * 96)

    for model_name, summary in models.items():
        avg = summary.get("multi_factor_averages") or {}
        vd = summary.get("verdict_distribution") or {}
        lat = summary.get("latency_stats") or {}
        counts_str = (
            f"{vd.get('CORRECT', 0)}/{vd.get('PARTIAL', 0)}/"
            f"{vd.get('INCORRECT', 0)}/{vd.get('ERROR', 0)}"
        )
        print(
            f"{model_name:<20} | "
            f"{summary.get('total_questions', 0):>4} | "
            f"{avg.get('overall_final_score', 0.0):>6.3f} | "
            f"{avg.get('factual_accuracy', 0.0):>6.3f} | "
            f"{avg.get('groundedness', 0.0):>6.3f} | "
            f"{avg.get('completeness', 0.0):>6.3f} | "
            f"{avg.get('adherence', 0.0):>6.3f} | "
            f"{counts_str:<17} | "
            f"{lat.get('avg_sec', 0.0):>6.2f}"
        )
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and score DeepInfra batch judge outputs")
    parser.add_argument("--batch-id", default="", help="Optional DeepInfra batch ID to check/download")
    parser.add_argument("--batch-output", default="bench/judge_batch_output.jsonl", help="Batch output JSONL path")
    parser.add_argument("--collected", default="bench/collected_answers.jsonl", help="Collected answers JSONL path")
    parser.add_argument("--output-dir", default="bench", help="Directory for per-model scored JSONL/summary files")
    parser.add_argument("--output-summary", default="bench/summary_batch.json", help="Combined summary JSON path")
    args = parser.parse_args()

    batch_output_path = Path(args.batch_output)
    collected_path = Path(args.collected)
    output_dir = Path(args.output_dir)
    output_summary_path = Path(args.output_summary)

    if args.batch_id.strip():
        ready = fetch_batch_output_if_ready(args.batch_id.strip(), batch_output_path)
        if not ready:
            sys.exit(0)

    if not batch_output_path.exists():
        print(f"Error: Batch output file {batch_output_path} not found.")
        sys.exit(1)

    if not collected_path.exists():
        print(f"Warning: Collected file {collected_path} not found; scoring without model latency/question metadata.")

    combined = score_batch_file(
        batch_output_path=batch_output_path,
        collected_path=collected_path,
        output_dir=output_dir,
        output_summary_path=output_summary_path,
    )
    print_comparison_table(combined)
    print(f"\nCombined summary saved to: {output_summary_path}")


if __name__ == "__main__":
    main()
