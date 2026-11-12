# Babel

Hybrid-searchable knowledge base system using `sqlite-vec` for vector retrieval, SQLite FTS5 for BM25 keyword search, and ZeroEntropy's `zerank-2` for reranking.

---

## Repository Structure

```
babel/
├── processing/                 # Offline / ETL data processing pipelines
│   ├── cleanup.py              # HTML to clean Markdown converter & boilerplate purging
│   ├── embeddings.py           # Document chunking & vector embedding pipeline (sqlite-vec)
│   └── summarize.py            # Programmatic date extraction, LLM summarizer & summary vectors
├── server/                     # Production server application package
│   ├── app.py                  # FastAPI app factory (create_app) & instance
│   ├── config.py               # Centralized settings and .env loader
│   ├── db.py                   # SQLite persistent connection pool with sqlite-vec
│   ├── models/                 # Pydantic schemas, ORM models, and text cleaners
│   │   ├── orm.py              # SQLAlchemy Document / StoredCrawlResult models
│   │   ├── schemas.py          # SearchRequest, SingleSearchResult, DocumentDetailResponse
│   │   └── text.py             # Normal text cleaner (strips markdown headers/links/images)
│   ├── routes/                 # API endpoint routers
│   │   ├── docs.py             # GET /docs/{doc_id} & /docs/{doc_id}/
│   │   ├── motd.py             # GET / (MOTD router)
│   │   └── search.py           # GET & POST /search & /api/search
│   └── services/               # Core business & search services
│       ├── embedding.py        # Runtime query embedding & scoring
│       ├── rerank.py           # ZeroEntropy zerank-2 candidate reranking
│       └── stash.py            # Unified Stash repository & search orchestrator
├── Containerfile               # Debian slim container specification
├── Dockerfile                  # Symlink to Containerfile
├── main.py                     # Server entrypoint (`uvicorn main:app`)
├── models.py                   # Backward-compatible facade for models
├── stash.py                    # Backward-compatible facade for Stash
├── embed_stash.py              # Backward-compatible facade for embeddings
├── rerank_stash.py             # Backward-compatible facade for reranker
├── process_cleanup.py          # Backward-compatible facade for cleanup CLI
└── summarize_stash.py          # Backward-compatible facade for summarizer CLI
```

---

## Running the Server

Start the API server locally with:

```sh
uvicorn main:app --host 127.0.0.1 --port 8000
```

### Endpoints

- **`GET /search?q={query}&limit=10&hybrid_blend=0.5&format=json|text`**
  Hybrid semantic vector (sqlite-vec) + keyword (FTS5 BM25) search with `zerank-2` reranking.
  Supports `format=text` or header `Accept: text/plain` for token-efficient RAG LLM context ingestion.
- **`POST /search`**
  Same search functionality with a JSON request body:
  ```json
  {"query": "học bổng", "limit": 10, "hybrid_blend": 0.5, "format": "json"}
  ```
- **`GET /docs/{doc_id}/?format=json|text&clean=true|false`**
  Fetch full document content and metadata by document ID.
- **`POST /docs/fetch`** (aliases: `/api/fetch`, `/docs/batch`)
  Batch fetch multiple documents in a single request:
  ```json
  {"doc_ids": [1525, 1102], "format": "text", "clean": true}
  ```
  Also available via `GET /docs/fetch?ids=1525,1102&format=text`.
- **`GET /`**
  Random Message of the Day (MOTD).

---

## Data Processing Pipelines

All data ETL and indexing pipelines live in `processing/`:

```sh
# 1. Clean raw crawled HTML and store normalized Markdown into stash.db
python -m processing.cleanup

# 2. Extract publication dates & generate LLM summaries
python -m processing.summarize --summarize

# 3. Generate vector embeddings for summaries
python -m processing.summarize --embed

# 4. Chunk documents and build vector embeddings in vec_chunks
python -m processing.embeddings --chunks
```

*(Root-level scripts `process_cleanup.py`, `summarize_stash.py`, and `embed_stash.py` are also preserved as backward-compatible CLI wrappers).*

---

## Container Deployment

### Using Docker Compose / Podman Compose (Recommended)

Start the entire stack with a single command:

```sh
docker compose up -d
# or with podman:
podman compose up -d
```

This reads `.env`, mounts `stash.db` into `/app/data/stash.db`, and exposes port `8000`.

### Manual Container Run

```sh
# Pull the pre-built image
docker pull ghcr.io/spikevn/ftu-babel:latest

# Run with mounted database files
docker run -d \
  --name babel \
  -p 8000:8000 \
  -v ./stash.db:/app/data/stash.db:rw \
  -v ./stash.db-wal:/app/data/stash.db-wal:rw \
  -v ./stash.db-shm:/app/data/stash.db-shm:rw \
  --env-file .env \
  ghcr.io/spikevn/ftu-babel:latest
```

---

## Search Reranking Configuration

Search retrieves top vector and keyword candidates, fuses them using Reciprocal Rank Fusion (RRF), then reranks them with ZeroEntropy's `zerank-2` model.

Configure in `.env`:

```sh
FTULABS_API_KEY="your_api_key"
FTULABS_API_ENDPOINT="https://your-api-host/v1"
```

Optional overrides:
- `ZEROENTROPY_API_KEY`: overrides `FTULABS_API_KEY`
- `ZEROENTROPY_BASE_URL`: overrides `FTULABS_API_ENDPOINT`
- `ZERANK_MODEL`: reranker model name (default: `zerank-2`)
- `ZERANK_CANDIDATE_K`: candidate count before reranking (default: `30`)
- `ZERANK_TIMEOUT`: request timeout in seconds (default: `30`)
