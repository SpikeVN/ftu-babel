#!/usr/bin/env python3
"""
Zero-cost local retrieval & reranking evaluator for Babel on FTU500 benchmark.
Evaluates:
- Recall@1, Recall@3, Recall@5, Recall@10 against ground truth sources
- MRR (Mean Reciprocal Rank)
- Ground Truth Answer Coverage in retrieved top passages
- Candidate context length and latency
- Zero DeepSeek API calls (100% local and budget-safe)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.services.stash import get_stash

def normalize_url(url: str) -> str:
    """Normalize URL by unquoting path and stripping trailing slashes."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path).rstrip("/")
        return f"{parsed.netloc.lower()}{path}"
    except Exception:
        return url.strip().lower()


def evaluate_retrieval(
    bench_file: str = "bench/ftu500.json",
    limit: int = 50,
    top_k: int = 10,
    hybrid_blend: float = 0.5,
) -> dict[str, Any]:
    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit > 0:
        data = data[:limit]

    stash = get_stash()

    total = len(data)
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    answer_covered_in_top1 = 0
    answer_covered_in_top5 = 0
    latencies = []
    rerank_text_lengths = []
    spam_detected_count = 0

    print(f"\nEvaluating Babel Retrieval & Reranking on {total} questions...")
    print(f"Parameters: top_k={top_k}, hybrid_blend={hybrid_blend}")
    print("-" * 65)

    for idx, item in enumerate(data, 1):
        qid = item["id"]
        question = item.get("question") or item.get("question_formal") or ""
        ground_truth_answer = (item.get("answer") or "").strip().lower()
        target_sources = [normalize_url(u) for u in (item.get("sources") or []) if u]

        t0 = time.time()
        try:
            results = stash.search_hybrid(
                query_text=question,
                top_k=top_k,
                hybrid_blend=hybrid_blend,
            )
        except Exception as exc:
            print(f"Error querying {qid}: {exc}")
            results = []
        elapsed = time.time() - t0
        latencies.append(elapsed)

        retrieved_urls = [normalize_url(r.get("url", "")) for r in results]
        
        # Check spam
        for u in retrieved_urls:
            if any(s in u for s in ["casimba", "casino", "baccarat", "kubet"]):
                spam_detected_count += 1

        # Check source recall
        rank_found = None
        for r_idx, r_url in enumerate(retrieved_urls, 1):
            if any(t in r_url or r_url in t for t in target_sources if t):
                rank_found = r_idx
                break

        if rank_found == 1:
            hits_at_1 += 1
        if rank_found and rank_found <= 3:
            hits_at_3 += 1
        if rank_found and rank_found <= 5:
            hits_at_5 += 1
        if rank_found and rank_found <= 10:
            hits_at_10 += 1

        if rank_found:
            reciprocal_ranks.append(1.0 / rank_found)
        else:
            reciprocal_ranks.append(0.0)

        # Check answer coverage in top retrieved passages
        if results:
            top1_text = f"{results[0].get('snippet', '')} {results[0].get('summary', '')}".lower()
            if ground_truth_answer and len(ground_truth_answer) > 2 and ground_truth_answer in top1_text:
                answer_covered_in_top1 += 1

            top5_text = " ".join(f"{r.get('snippet', '')} {r.get('summary', '')}" for r in results[:5]).lower()
            if ground_truth_answer and len(ground_truth_answer) > 2 and ground_truth_answer in top5_text:
                answer_covered_in_top5 += 1

        status_mark = f"Rank #{rank_found}" if rank_found else "Miss"
        print(f"[{idx:03d}/{total:03d}] {qid} | {status_mark:8s} | {elapsed:.2f}s | {question[:45]}")

    mrr = sum(reciprocal_ranks) / total if total > 0 else 0.0
    r1 = hits_at_1 / total if total > 0 else 0.0
    r3 = hits_at_3 / total if total > 0 else 0.0
    r5 = hits_at_5 / total if total > 0 else 0.0
    r10 = hits_at_10 / total if total > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    summary = {
        "total_evaluated": total,
        "mrr": round(mrr, 4),
        "recall_at_1": round(r1, 4),
        "recall_at_3": round(r3, 4),
        "recall_at_5": round(r5, 4),
        "recall_at_10": round(r10, 4),
        "answer_coverage_in_top1": round(answer_covered_in_top1 / total, 4) if total else 0.0,
        "answer_coverage_in_top5": round(answer_covered_in_top5 / total, 4) if total else 0.0,
        "spam_hits_detected": spam_detected_count,
        "avg_latency_sec": round(avg_latency, 2),
    }

    print("\n" + "=" * 65)
    print("           RETRIEVAL & RERANKING EVALUATION RESULTS")
    print("=" * 65)
    print(f"Total Questions Evaluated:  {summary['total_evaluated']}")
    print(f"Mean Reciprocal Rank (MRR): {summary['mrr']:.4f}")
    print(f"Recall@1:                   {summary['recall_at_1'] * 100:.1f}%")
    print(f"Recall@3:                   {summary['recall_at_3'] * 100:.1f}%")
    print(f"Recall@5:                   {summary['recall_at_5'] * 100:.1f}%")
    print(f"Recall@10:                  {summary['recall_at_10'] * 100:.1f}%")
    print(f"Answer Coverage (Top 1):    {summary['answer_coverage_in_top1'] * 100:.1f}%")
    print(f"Answer Coverage (Top 5):    {summary['answer_coverage_in_top5'] * 100:.1f}%")
    print(f"Spam Infiltrations Found:   {summary['spam_hits_detected']}")
    print(f"Average Latency:            {summary['avg_latency_sec']}s")
    print("=" * 65)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval accuracy on FTU500")
    parser.add_argument("--limit", type=int, default=20, help="Number of questions to test (default: 20)")
    parser.add_argument("--top-k", type=int, default=10, help="Top K results (default: 10)")
    parser.add_argument("--blend", type=float, default=0.5, help="Hybrid blend (default: 0.5)")
    args = parser.parse_args()

    evaluate_retrieval(limit=args.limit, top_k=args.top_k, hybrid_blend=args.blend)
