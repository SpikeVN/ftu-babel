"""Vector Embedding Pipeline & Scoring with sqlite-vec (backward-compatibility wrapper)."""

import sys
from processing.embeddings import (
    CHUNK_CHARS,
    CHUNK_FANOUT,
    CHUNK_OVERLAP,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    DOC_PREFIX,
    MAX_CHAR_LENGTH,
    MAX_EMBED_CHARS,
    MAX_RETRIES,
    QUERY_PREFIX,
    VECTOR_DIM,
    init_chunk_tables,
    init_sqlite_vec_tables,
    main,
    prepare_embedding_text,
    run_chunk_embedding,
    run_embedding_for_all_documents,
    split_into_chunks,
)
from server.config import (
    DEFAULT_DB_PATH,
    DEFAULT_EMBED_API_KEY as DEFAULT_API_KEY,
    DEFAULT_EMBED_URL,
    FTULABS_API_ENDPOINT,
    load_env_file,
)
from server.services.embedding import (
    EmbeddingClient,
    get_shared_embed_client,
    score_vector,
    search_similar,
)

__all__ = [
    "load_env_file",
    "FTULABS_API_ENDPOINT",
    "DEFAULT_EMBED_URL",
    "DEFAULT_MODEL",
    "DEFAULT_API_KEY",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHUNK_BATCH_SIZE",
    "DEFAULT_CONCURRENCY",
    "MAX_RETRIES",
    "VECTOR_DIM",
    "MAX_CHAR_LENGTH",
    "CHUNK_CHARS",
    "CHUNK_OVERLAP",
    "MAX_EMBED_CHARS",
    "CHUNK_FANOUT",
    "QUERY_PREFIX",
    "DOC_PREFIX",
    "prepare_embedding_text",
    "split_into_chunks",
    "EmbeddingClient",
    "init_sqlite_vec_tables",
    "init_chunk_tables",
    "run_embedding_for_all_documents",
    "run_chunk_embedding",
    "get_shared_embed_client",
    "score_vector",
    "search_similar",
    "main",
]

if __name__ == "__main__":
    main()
