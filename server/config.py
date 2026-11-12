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

# FTULabs Gateway / Upstream API
FTULABS_API_ENDPOINT = os.environ.get("FTULABS_API_ENDPOINT", "").rstrip("/")
FTULABS_API_KEY = os.environ.get("FTULABS_API_KEY", "")


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
    _join_endpoint(FTULABS_API_ENDPOINT, "embeddings")
    if FTULABS_API_ENDPOINT
    else "http://127.0.0.1:8888/v1/embeddings"
)
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "unsloth/embeddinggemma-2-GGUF")
DEFAULT_EMBED_API_KEY = os.environ.get("EMBED_API_KEY") or FTULABS_API_KEY
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

# ZeroEntropy Reranker Settings
ZEROENTROPY_API_KEY = os.environ.get("ZEROENTROPY_API_KEY") or FTULABS_API_KEY
ZEROENTROPY_BASE_URL = (
    os.environ.get("ZEROENTROPY_BASE_URL")
    or FTULABS_API_ENDPOINT
    or "https://api.zeroentropy.dev/v1"
).rstrip("/")
ZERANK_MODEL = os.environ.get("ZERANK_MODEL", "zerank-2")
ZERANK_CANDIDATE_K = int(os.environ.get("ZERANK_CANDIDATE_K", "30"))
ZERANK_TIMEOUT = float(os.environ.get("ZERANK_TIMEOUT", "30"))

# LLM Summarization Settings
DEFAULT_LLM_URL = os.environ.get("LLM_URL") or (
    _join_endpoint(FTULABS_API_ENDPOINT, "chat/completions")
    if FTULABS_API_ENDPOINT
    else "http://127.0.0.1:8888/v1/chat/completions"
)
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "unsloth/gemma-4-E2B-it-qat-GGUF")
DEFAULT_LLM_API_KEY = os.environ.get("LLM_API_KEY") or FTULABS_API_KEY
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "3000"))
