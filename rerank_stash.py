"""Rerank search candidates using zerank-2 (re-exported from server.services.rerank and server.config)."""

from server.config import (
    FTULABS_API_ENDPOINT,
    FTULABS_API_KEY,
    MOSS_API_KEY,
    MOSS_EMBEDDING_ENDPOINT,
    MOSS_RERANKING_ENDPOINT,
    ZERANK_CANDIDATE_K,
    ZERANK_MODEL,
    ZERANK_TIMEOUT,
    ZEROENTROPY_API_KEY,
    ZEROENTROPY_BASE_URL,
)
from server.services.rerank import (
    get_shared_rerank_client,
    rerank_hits,
)

__all__ = [
    "MOSS_API_KEY",
    "MOSS_EMBEDDING_ENDPOINT",
    "MOSS_RERANKING_ENDPOINT",
    "FTULABS_API_ENDPOINT",
    "FTULABS_API_KEY",
    "ZEROENTROPY_API_KEY",
    "ZEROENTROPY_BASE_URL",
    "ZERANK_MODEL",
    "ZERANK_CANDIDATE_K",
    "ZERANK_TIMEOUT",
    "get_shared_rerank_client",
    "rerank_hits",
]
