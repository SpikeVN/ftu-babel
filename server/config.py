"""Centralized configuration and environment settings for Babel."""

from __future__ import annotations

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path: Path | None = None) -> None:
    """Minimal .env loader: sets KEY=VALUE pairs without overriding existing environment variables."""
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.exists():
        return
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                os.environ.setdefault(key, value)
    except OSError:
        pass


# Automatically load .env on module import
load_env_file()

# Database settings
_env_db_path = os.environ.get("STASH_DB_PATH")
if _env_db_path:
    DEFAULT_DB_PATH = _env_db_path
elif Path("/app/data/stash.db").exists():
    DEFAULT_DB_PATH = "/app/data/stash.db"
else:
    DEFAULT_DB_PATH = str(PROJECT_ROOT / "stash.db")

_env_usearch_path = os.environ.get("USEARCH_INDEX_PATH")
if _env_usearch_path:
    DEFAULT_USEARCH_PATH = _env_usearch_path
elif Path("/app/data/stash.usearch").exists():
    DEFAULT_USEARCH_PATH = "/app/data/stash.usearch"
else:
    DEFAULT_USEARCH_PATH = str(PROJECT_ROOT / "stash.usearch")

# MOSS Gateway / Upstream API
MOSS_API_KEY = os.environ.get("MOSS_API_KEY") or os.environ.get("FTULABS_API_KEY", "")
MOSS_EMBEDDING_ENDPOINT = (
    os.environ.get("MOSS_EMBEDDING_ENDPOINT")
    or os.environ.get("FTULABS_API_ENDPOINT", "")
).rstrip("/")
MOSS_RERANKING_ENDPOINT = (
    os.environ.get("MOSS_RERANKING_ENDPOINT")
    or os.environ.get("FTULABS_API_ENDPOINT", "")
).rstrip("/")

# Backwards-compatibility aliases
FTULABS_API_KEY = MOSS_API_KEY
FTULABS_API_ENDPOINT = MOSS_EMBEDDING_ENDPOINT

# SearXNG Web Search
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080").rstrip("/")


def _is_local_endpoint(url: str) -> bool:
    return any(host in url for host in ("127.0.0.1", "localhost", "host.docker.internal"))


def _join_endpoint(base: str, path: str) -> str:
    base = base.rstrip("/")
    path = path.lstrip("/")
    if base.endswith("/v1") and path.startswith("v1/"):
        path = path[3:]
    elif not base.endswith("/v1") and not path.startswith("v1/"):
        path = f"v1/{path}"
    return f"{base}/{path}" if base else ""


# Embedding Settings
DEFAULT_EMBED_URL = os.environ.get("EMBED_URL") or (
    _join_endpoint(MOSS_EMBEDDING_ENDPOINT, "embeddings")
    if MOSS_EMBEDDING_ENDPOINT
    else "http://127.0.0.1:8888/v1/embeddings"
)
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "unsloth/embeddinggemma-2-GGUF")
_env_embed_key = os.environ.get("EMBED_API_KEY")
if _env_embed_key is not None:
    DEFAULT_EMBED_API_KEY = _env_embed_key
elif _is_local_endpoint(DEFAULT_EMBED_URL):
    DEFAULT_EMBED_API_KEY = ""
else:
    DEFAULT_EMBED_API_KEY = MOSS_API_KEY
DEFAULT_EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
DEFAULT_CHUNK_BATCH_SIZE = int(os.environ.get("EMBED_CHUNK_BATCH_SIZE", "16"))
DEFAULT_CONCURRENCY = int(os.environ.get("EMBED_CONCURRENCY", "4"))
MAX_RETRIES = int(os.environ.get("EMBED_MAX_RETRIES", "6"))

CHUNK_CHARS = int(os.environ.get("EMBED_CHUNK_CHARS", "4000"))
CHUNK_OVERLAP = int(os.environ.get("EMBED_CHUNK_OVERLAP", "400"))
MAX_EMBED_CHARS = int(os.environ.get("EMBED_MAX_CHARS", str(CHUNK_CHARS + CHUNK_OVERLAP)))
CHUNK_FANOUT = int(os.environ.get("EMBED_CHUNK_FANOUT", "5"))

QUERY_PREFIX = os.environ.get("EMBED_QUERY_PREFIX", "task: search result | query: ")
DOC_PREFIX = os.environ.get("EMBED_DOC_PREFIX", "title: none | text: ")

# ZeroEntropy / MOSS Reranker Settings
ZEROENTROPY_API_KEY = os.environ.get("ZEROENTROPY_API_KEY") or MOSS_API_KEY
ZEROENTROPY_BASE_URL = (
    os.environ.get("ZEROENTROPY_BASE_URL")
    or (
        MOSS_RERANKING_ENDPOINT
        if (MOSS_RERANKING_ENDPOINT.endswith("/v1") or not MOSS_RERANKING_ENDPOINT)
        else f"{MOSS_RERANKING_ENDPOINT}/v1"
    )
    or "https://api.zeroentropy.dev/v1"
).rstrip("/")
ZERANK_MODEL = os.environ.get("ZERANK_MODEL", "zerank-2")
ZERANK_CANDIDATE_K = int(os.environ.get("ZERANK_CANDIDATE_K", "30"))
ZERANK_TIMEOUT = float(os.environ.get("ZERANK_TIMEOUT", "45"))
RERANK_ENABLED = bool(ZEROENTROPY_API_KEY or _is_local_endpoint(ZEROENTROPY_BASE_URL))

# LLM Summarization Settings
_llm_base = MOSS_EMBEDDING_ENDPOINT or MOSS_RERANKING_ENDPOINT
DEFAULT_LLM_URL = os.environ.get("LLM_URL") or (
    _join_endpoint(_llm_base, "chat/completions")
    if _llm_base
    else "http://127.0.0.1:8888/v1/chat/completions"
)
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "unsloth/gemma-4-E2B-it-qat-GGUF")
DEFAULT_LLM_API_KEY = os.environ.get("LLM_API_KEY") or MOSS_API_KEY
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "3000"))
