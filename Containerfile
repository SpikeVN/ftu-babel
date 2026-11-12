# Babel - Hybrid Search & Knowledge Base Containerfile
FROM python:3.12-slim AS runtime

# Prevent Python from writing .pyc files & buffer output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install essential runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and app directory
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy dependency specifications first for layer caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "html2text>=2024.2.26" \
    "httpx>=0.27.0" \
    "sqlalchemy>=2.0.0" \
    "sqlite-vec>=0.1.9" \
    "tqdm>=4.66.0" \
    "uvicorn>=0.30.0"

# Copy application source code
COPY --chown=appuser:appuser server/ /app/server/
COPY --chown=appuser:appuser processing/ /app/processing/
COPY --chown=appuser:appuser modules/ /app/modules/
COPY --chown=appuser:appuser main.py models.py stash.py embed_stash.py rerank_stash.py process_cleanup.py summarize_stash.py /app/

# Switch to non-root user
USER appuser

# Expose server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
