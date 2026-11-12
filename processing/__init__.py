"""Data processing and ETL pipelines for Babel.

Contains pipelines for:
  - cleanup: Cleaning raw crawled HTML, converting to clean markdown, purging footers and boilerplates.
  - summarize: Unified date extraction, LLM summarization, and summary vector indexing.
  - embeddings: Document and chunk-level vector embedding pipeline using sqlite-vec.
"""

from processing.cleanup import clean_database, clean_html, clean_markdown
from processing.embeddings import (
    init_chunk_tables,
    init_sqlite_vec_tables,
    run_chunk_embedding,
    run_embedding_for_all_documents,
    split_into_chunks,
)
from processing.summarize import (
    extract_programmatic_date,
    init_summary_tables,
    run_summarization,
    run_summary_embeddings,
)

__all__ = [
    "clean_database",
    "clean_html",
    "clean_markdown",
    "init_chunk_tables",
    "init_sqlite_vec_tables",
    "run_chunk_embedding",
    "run_embedding_for_all_documents",
    "split_into_chunks",
    "extract_programmatic_date",
    "init_summary_tables",
    "run_summarization",
    "run_summary_embeddings",
]
