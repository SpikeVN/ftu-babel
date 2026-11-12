"""Unified Document Summarization & Vector Indexing (backward-compatibility wrapper delegating to processing.summarize)."""

import sys
from processing.summarize import (
    DOC_PREFIX,
    MAX_PROMPT_CHARS,
    MAX_RETRIES,
    QUERY_PREFIX,
    VECTOR_DIM,
    extract_programmatic_date,
    get_vec_connection,
    init_summary_tables,
    load_env_file,
    main,
    run_summarization,
    run_summary_embeddings,
    search_summaries,
)
from server.config import (
    DEFAULT_EMBED_API_KEY,
    DEFAULT_EMBED_MODEL,
    DEFAULT_EMBED_URL,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_URL,
)

__all__ = [
    "load_env_file",
    "DEFAULT_LLM_URL",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_LLM_API_KEY",
    "DEFAULT_EMBED_URL",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_EMBED_API_KEY",
    "VECTOR_DIM",
    "DOC_PREFIX",
    "QUERY_PREFIX",
    "MAX_PROMPT_CHARS",
    "MAX_RETRIES",
    "get_vec_connection",
    "init_summary_tables",
    "extract_programmatic_date",
    "run_summarization",
    "run_summary_embeddings",
    "search_summaries",
    "main",
]

if __name__ == "__main__":
    main()
