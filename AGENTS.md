# AGENTS.md — Babel Knowledge Base Operations & Deployment Guide

This document describes the codebase conventions, scraping/OCR/embedding pipeline, and deployment workflow for **Babel** (`SpikeVN/ftu-babel`).

## 1. Environment & Configuration

- **Python environment**: Use `.venv/bin/python` (or `uv run python`).
- **Centralized config**: [`server/config.py`](server/config.py) loads `.env` at startup via `load_env_file()` without overriding existing process environment variables.
- **Key environment variables**:
  - `MOSS_API_KEY`: API key for remote MOSS gateway (optional when endpoints target `localhost`, `127.0.0.1`, or `host.docker.internal`).
  - `MOSS_EMBEDDING_ENDPOINT`: Base URL for embeddings (e.g. `https://550w.falcolabs.org/v1/` or `http://localhost:8086/v1/`).
  - `MOSS_RERANKING_ENDPOINT`: Base URL for reranking (e.g. `https://550w.falcolabs.org/v1/` or `http://localhost:8012/v1/`).
  - `STASH_DB_PATH`: Path to SQLite database (defaults to `/app/data/stash.db` in Docker or `./stash.db` locally).
  - `USEARCH_INDEX_PATH`: Path to USearch HNSW index (defaults to `/app/data/stash.usearch` in Docker or `./stash.usearch` locally).
  - Backwards-compatible aliases `FTULABS_API_KEY` and `FTULABS_API_ENDPOINT` are still recognized by `server/config.py`.

## 2. Scraping, Extraction & Processing Pipeline (`processing/`)

All data ingestion, parsing, OCR, cleanup, and embedding scripts live under `processing/`.

### Unified Periodic Pipeline (`processing/pipeline.py`)

Run `--status` to inspect current counts and pending work, `--all` to execute all stages idempotently (skipping already-processed items), or target individual stages:

```bash
.venv/bin/python processing/pipeline.py --status
.venv/bin/python processing/pipeline.py --all
```

Stages and flags:
1. **Link Canonicalization & Stubbing (`--links`)**
   - `processing/populate_document_stubs.py`, `processing/migrate_links_to_docids.py`
   - Normalizes FTU links/file references and replaces them with internal `[Link](/docs/{id})` references.
2. **Stub Fetching (`--fetch`)**
   - `processing/fetch_pdfs.py`, `processing/fetch_other_files.py`, `processing/resolve_stubs_mime.py`
   - Downloads pending document stubs and resolves MIME types.
3. **Document Parsing & OCR (`--parse`, `--concurrency-parse N`)**
   - `processing/parse_all_documents.py`, `processing/extract_pdfs.py`, `processing/process_remaining_unextracted.py`
   - Extracts text from PDFs, Office files, and HTML, running OCR where needed.
4. **Embedded Image Filtering & Diagram Processing (`--images`)**
   - `processing/process_embedded_images.py`, `processing/finalize_stage2_injection.py`
   - Filters boilerplate/template images and runs OCR/VLM extraction on content images/diagrams.
5. **Hierarchical Chunking & Embedding (`--embed`, `--concurrency-embed N`)**
   - `processing/embeddings.py`, `processing/build_usearch_index.py`
   - Splits documents into parent (`document_parents`) and child (`vec_chunks`) chunks, embeds via `MOSS_EMBEDDING_ENDPOINT`, and updates both `stash.db` and `stash.usearch`.
6. **Export Unscraped Links (`--export-unscraped`)**
   - Exports remaining unscraped links to `unscraped_ftu_links.csv` / `processing/unscraped_links.csv` for future scraping runs.

Cleanup & summarization helpers:
- `processing/clean_spam.py`, `processing/cleanup.py`, `processing/summarize.py`.

## 3. Embedding, Search & Database Maintenance

- **Hybrid Retrieval** (`server/services/stash.py` → `Stash.search_hybrid`):
  1. Vector search via `stash.usearch` (`USEARCH_INDEX_PATH`) over child chunks, mapped back to parent context in `stash.db`.
  2. Keyword search via SQLite FTS5 (`documents_fts`).
  3. Reciprocal Rank Fusion (RRF) blending (`hybrid_blend`).
  4. Cross-encoder reranking via `server/services/rerank.py` (`MOSS_RERANKING_ENDPOINT`). When the rerank endpoint is local (`localhost`, `127.0.0.1`, `host.docker.internal`), `RERANK_ENABLED` is `True` without requiring `MOSS_API_KEY`.
- **Rebuilding the USearch Index** from `stash.db`:
  ```bash
  .venv/bin/python processing/build_usearch_index.py
  ```
- **Vacuuming `stash.db`**:
  Because `stash.db` contains `sqlite-vec` (`vec0`) virtual tables, always load `sqlite_vec` before running `VACUUM`:
  ```python
  import sqlite3, sqlite_vec
  conn = sqlite3.connect("stash.db")
  conn.enable_load_extension(True)
  sqlite_vec.load(conn)
  conn.enable_load_extension(False)
  conn.execute("VACUUM")
  conn.close()
  ```

## 4. Deployment Process (`GitHub Actions` + `triph@xavier`)

### 4.1 Container Image Build
- Pushing to `main` (or `v*` tags) triggers [`.github/workflows/build-arm64.yml`](.github/workflows/build-arm64.yml) on `ubuntu-24.04-arm`.
- It builds [`Containerfile`](Containerfile) for `linux/arm64` and pushes `ghcr.io/spikevn/ftu-babel:latest` (plus `:latest-arm64`, `:arm64`, and `:<short-sha>`).

### 4.2 Remote Host Setup (`triph@xavier:/mnt/ssd/babel`)
- Remote `/mnt/ssd/babel/.env` contains **no API keys** and points to local host services:
  ```dotenv
  MOSS_EMBEDDING_ENDPOINT=http://localhost:8086/v1/
  MOSS_RERANKING_ENDPOINT=http://localhost:8012/v1/
  ```
- **Sync codebase** (do not use `--delete-excluded`, as `searxng/` is owned by container UID `977`):
  ```bash
  rsync -av \
    --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
    --exclude='.env' --exclude='downloads/' --exclude='data/' \
    --exclude='*.db*' --exclude='*.sqlite*' --exclude='*.usearch' \
    --exclude='*.zst' --exclude='searxng/' \
    --exclude='bench/*.jsonl' --include='bench/ftu500.json' --exclude='bench/*.json' \
    ./ triph@xavier:/mnt/ssd/babel/
  ```
- **Sync `stash.db` and `stash.usearch` (`zstd` + `rsync`)**:
  ```bash
  zstd -T0 -f stash.db -o stash.db.zst
  zstd -T0 -f stash.usearch -o stash.usearch.zst
  rsync -av --progress stash.db.zst stash.usearch.zst triph@xavier:/mnt/ssd/babel/
  ssh triph@xavier "cd /mnt/ssd/babel && \
    unzstd -f --rm stash.db.zst -o stash.db.new && \
    unzstd -f --rm stash.usearch.zst -o stash.usearch.new && \
    chmod 666 stash.db.new stash.usearch.new && \
    mv -f stash.db.new stash.db && \
    mv -f stash.usearch.new stash.usearch"
  rm -f stash.db.zst stash.usearch.zst
  ```
- **Restart Remote Service** (after GitHub Actions finishes pushing the updated image):
  ```bash
  ssh triph@xavier "cd /mnt/ssd/babel && docker compose pull && docker compose up -d"
  ```
