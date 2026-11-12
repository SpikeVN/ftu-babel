"""Rerank vector and hybrid search candidates with ZeroEntropy's zerank-2 model."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from server.config import (
    ZEROENTROPY_API_KEY,
    ZEROENTROPY_BASE_URL,
    ZERANK_CANDIDATE_K,
    ZERANK_MODEL,
    ZERANK_TIMEOUT,
)

logger = logging.getLogger(__name__)

_shared_rerank_client: httpx.Client | None = None


def get_shared_rerank_client() -> httpx.Client:
    """Retrieve or create a persistent keep-alive HTTP client for reranking."""
    global _shared_rerank_client
    if _shared_rerank_client is None or _shared_rerank_client.is_closed:
        _shared_rerank_client = httpx.Client(
            timeout=ZERANK_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=120.0),
        )
    return _shared_rerank_client


def _rerank_with_completion(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
) -> list[dict[str, Any]]:
    """Rerank candidates using an OpenAI-compatible completion endpoint."""
    candidates_to_rank = hits[:min(len(hits), 15)]
    doc_entries = []
    id_map = {}
    for idx, hit in enumerate(candidates_to_rank, 1):
        id_map[idx] = hit
        title = hit.get("title") or ""
        snippet = hit.get("snippet") or hit.get("rerank_text") or ""
        text = f"{title} - {snippet}" if title else snippet
        doc_entries.append(f"[{idx}] {text[:250].strip()}")

    prompt = f"""Bạn là bộ máy xếp hạng tìm kiếm (reranker).
Nhiệm vụ: Đánh giá mức độ liên quan của các tài liệu ứng viên với câu truy vấn.
Chỉ trả về danh sách JSON các ID sắp xếp theo thứ tự giảm dần từ phù hợp nhất đến ít phù hợp nhất.
Ví dụ: [1, 3, 2, ...]
Tuyệt đối không giải thích, không viết thêm bất cứ chữ nào khác ngoài mảng JSON.

Truy vấn: {query}

Ứng viên:
""" + "\n".join(doc_entries) + "\n\nJSON:"

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    client = get_shared_rerank_client()
    resp = client.post(
        f"{base_url}/completions",
        headers=headers,
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": 60,
            "temperature": 0.0,
            "stop": ["]"],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    raw_text = resp.json()["choices"][0]["text"] + "]"
    match = re.search(r"\[[0-9\s,]+\]", raw_text)
    if not match:
        raise ValueError(f"Could not parse ranked IDs from zerank-2 completion: {raw_text}")
    ranked_indices = json.loads(match.group(0))

    reranked: list[dict[str, Any]] = []
    seen: set[int] = set()
    total = len(ranked_indices)
    for rank_pos, idx in enumerate(ranked_indices):
        if idx in id_map and idx not in seen:
            seen.add(idx)
            item = dict(id_map[idx])
            item.pop("rerank_text", None)
            item["score"] = round(1.0 - (rank_pos * 0.5 / max(total, 1)), 4)
            reranked.append(item)

    for idx, hit in id_map.items():
        if idx not in seen:
            item = dict(hit)
            item.pop("rerank_text", None)
            item["score"] = 0.1
            reranked.append(item)

    # Append any candidates beyond top 15
    for hit in hits[len(candidates_to_rank):]:
        item = dict(hit)
        item.pop("rerank_text", None)
        reranked.append(item)

    return reranked[:top_k]


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int,
    api_key: str = ZEROENTROPY_API_KEY,
    base_url: str = ZEROENTROPY_BASE_URL,
    model: str = ZERANK_MODEL,
    timeout: float = ZERANK_TIMEOUT,
) -> list[dict[str, Any]]:
    """Reorder search candidates by relevance using zerank-2."""
    if not hits:
        return []
    if not api_key:
        raise ValueError(
            "Set FTULABS_API_KEY or ZEROENTROPY_API_KEY to rerank search results."
        )

    candidates = hits[:min(len(hits), max(top_k, 12))]
    documents = [
        str(hit.get("summary") or hit.get("rerank_text") or hit.get("snippet") or "")[:600]
        for hit in candidates
    ]

    endpoints_to_try = [
        f"{base_url}/rerank",
        f"{base_url}/models/rerank",
    ]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(documents)),
    }

    last_error: Exception | None = None
    client = get_shared_rerank_client()
    for endpoint in endpoints_to_try:
        try:
            response = client.post(endpoint, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            data = response.json()
            results = data.get("results")
            if not isinstance(results, list):
                continue

            ranked: list[tuple[float, dict[str, Any]]] = []
            seen_indices: set[int] = set()
            for result in results:
                if not isinstance(result, dict):
                    continue
                index = result.get("index")
                score = result.get("relevance_score")
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or index < 0
                    or index >= len(candidates)
                    or index in seen_indices
                    or not isinstance(score, (int, float))
                    or isinstance(score, bool)
                ):
                    continue
                seen_indices.add(index)
                hit = dict(candidates[index])
                hit.pop("rerank_text", None)
                hit["score"] = float(score)
                ranked.append((float(score), hit))

            ranked.sort(key=lambda item: item[0], reverse=True)
            reranked_hits = [hit for _, hit in ranked]

            # Append candidates that weren't scored
            for idx, cand in enumerate(candidates):
                if idx not in seen_indices:
                    item = dict(cand)
                    item.pop("rerank_text", None)
                    reranked_hits.append(item)

            # Append hits beyond candidates
            for hit in hits[len(candidates):]:
                item = dict(hit)
                item.pop("rerank_text", None)
                reranked_hits.append(item)

            return reranked_hits[:top_k]
        except Exception as exc:
            last_error = exc
            logger.info("Rerank attempt at %s failed: %s", endpoint, exc)

    logger.warning("Native rerank endpoints failed (%s); trying completion-based ranking.", last_error)
    return _rerank_with_completion(
        query,
        hits,
        top_k=top_k,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
    )
