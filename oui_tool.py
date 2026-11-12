"""OpenWebUI Tool: FTU Babel Knowledge Base Retrieval (babel_query & babel_fetch).

Provides integration with the Babel hybrid knowledge-base search engine:
  - babel_query: Hybrid / semantic / keyword search with zerank-2 reranking.
  - babel_fetch: Full-document batch retrieval by document ID(s) in a single request.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Sequence, Union

from pydantic import BaseModel, Field

# Try importing requests if available in OpenWebUI environment
try:
    import requests
except ImportError:
    requests = None  # type: ignore

DEFAULT_API_URL = os.environ.get(
    "SEARCH_API_URL", "https://babel.falcolabs.org"
)
DEFAULT_API_KEY = os.environ.get("SEARCH_API_KEY", "")


class Tools:
    class Valves(BaseModel):
        SEARCH_API_URL: str = Field(
            default=DEFAULT_API_URL,
            description="Base URL for the Babel API service (e.g. https://babel.falcolabs.org)",
        )
        SEARCH_API_KEY: str = Field(
            default=DEFAULT_API_KEY,
            description="Optional API key for authorization (Bearer token)",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    def _config(self) -> "Valves":
        """Retrieve tool configuration valves safely."""
        valves = getattr(self, "valves", None)
        if valves is None:
            valves = self.Valves()
        return valves

    def _base_url(self) -> str:
        """Resolve the base URL by stripping trailing path suffixes like /api/search or /search."""
        url = self._config().SEARCH_API_URL.strip().rstrip("/")
        for suffix in ("/api/search", "/search", "/api/fetch", "/docs", "/api"):
            if url.endswith(suffix):
                url = url[: -len(suffix)]
                break
        return url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        """Construct standard HTTP headers."""
        headers = {
            "Accept": "text/plain, application/json",
            "Content-Type": "application/json",
            "User-Agent": "OpenWebUI-BabelTool/1.0",
        }
        key = self._config().SEARCH_API_KEY.strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _http_post(self, url: str, json_data: dict[str, Any], timeout: float = 60.0) -> tuple[int, str]:
        """Execute HTTP POST request with requests or fallback to urllib."""
        headers = self._headers()
        if requests is not None:
            resp = requests.post(url, json=json_data, headers=headers, timeout=timeout)
            return resp.status_code, resp.text

        data_bytes = json.dumps(json_data).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status, response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", errors="replace")

    def babel_query(
        self,
        query: str,
        mode: str = "semantic",
        top_k: int = 10,
        format: str = "text",
    ) -> str:
        """
        Search the FTU (Trường Đại học Ngoại thương) knowledge base.
        Returns relevant document passages with their doc_id, title, URL, date, and score.

        Args:
            query: The search query string (use user's literal wording, typos included).
            mode: Search mode:
                  - 'semantic' (default): Vector search for general topics, concepts, and questions.
                  - 'keyword': BM25 keyword search for person names, acronyms, codes, or course IDs
                    (e.g., KTQT301, FTDS, QLĐT).
                  - 'hybrid': 50/50 balanced blend of semantic and keyword search.
            top_k: Maximum number of passage results to return (1-50, default 10).
            format: Output format, either 'text' (default, citation-ready plain text for LLM) or 'json'.

        Returns:
            Citation-ready passages with doc_id, URL, publication date, summary, and clean text content.
        """
        base_url = self._base_url()
        endpoint = f"{base_url}/query"

        # Map system prompt search modes to server hybrid_blend
        # hybrid_blend: 0.0 = pure vector (semantic), 1.0 = pure keyword (BM25), 0.5 = hybrid
        mode_lower = (mode or "semantic").strip().lower()
        if mode_lower == "keyword":
            blend = 1.0
        elif mode_lower == "semantic":
            blend = 0.0
        else:
            blend = 0.5

        limit = max(1, min(int(top_k), 50))
        payload = {
            "query": query,
            "limit": limit,
            "hybrid_blend": blend,
            "format": format if format in ("text", "json") else "text",
        }

        try:
            status_code, body = self._http_post(endpoint, payload, timeout=60.0)
            if status_code == 404:
                # Fallback to legacy /search endpoint if /query not yet available
                status_code, body = self._http_post(f"{base_url}/search", payload, timeout=60.0)
            if status_code >= 400:
                return f"Error {status_code} querying Babel knowledge base: {body}"
            return body
        except Exception as exc:
            return f"Error querying Babel knowledge base: {exc}"

    def web_search(
        self,
        query: str,
        limit: int = 5,
        format: str = "text",
    ) -> str:
        """
        Search the public web for real-time, external, or non-FTU information via SearXNG.
        Use as a fallback when FTU knowledge base (babel_query) does not contain the answer.

        Args:
            query: The web search query string.
            limit: Maximum number of search results to return (1-20, default 5).
            format: Output format, either 'text' (default) or 'json'.

        Returns:
            Web search result snippets with titles and URLs.
        """
        base_url = self._base_url()
        endpoint = f"{base_url}/web_search"
        payload = {
            "query": query,
            "limit": max(1, min(int(limit), 20)),
            "format": format if format in ("text", "json") else "text",
        }
        try:
            status_code, body = self._http_post(endpoint, payload, timeout=30.0)
            if status_code >= 400:
                return f"Error {status_code} performing web search: {body}"
            return body
        except Exception as exc:
            return f"Error performing web search: {exc}"

    def babel_fetch(
        self,
        doc_id: Union[int, str, list[Union[int, str]]] = None,
        ids: Union[list[Union[int, str]], int, str] = None,
        format: str = "text",
    ) -> str:
        """
        Fetch the full document content and metadata for one or more doc_ids in a single batch request.
        Use when search passages in babel_query lack specific details, or when full text context is needed.

        Args:
            doc_id: A single document ID (e.g., 1525) or list of doc_ids returned by babel_query.
            ids: Optional alias for doc_id (accepts a single ID or list of IDs, e.g. [1525, 1526]).
            format: Output format, either 'text' (default, clean plain text) or 'json'.

        Returns:
            Full clean text content of the document(s) along with Title, URL, Date, and Summary.
        """
        # Collect and normalize all requested IDs
        target_ids: list[int] = []
        for raw in (doc_id, ids):
            if raw is None:
                continue
            if isinstance(raw, (int, str)):
                for part in str(raw).split(","):
                    part = part.strip()
                    if part.isdigit():
                        target_ids.append(int(part))
            elif isinstance(raw, (list, tuple, set)):
                for item in raw:
                    if isinstance(item, int):
                        target_ids.append(item)
                    elif isinstance(item, str) and item.strip().isdigit():
                        target_ids.append(int(item.strip()))

        # Deduplicate while preserving order
        target_ids = list(dict.fromkeys(target_ids))

        if not target_ids:
            return "Error: No valid numeric doc_id provided to babel_fetch."

        base_url = self._base_url()
        endpoint = f"{base_url}/docs/fetch"
        payload = {
            "doc_ids": target_ids,
            "clean": True,
            "format": format if format in ("text", "json") else "text",
        }

        try:
            status_code, body = self._http_post(endpoint, payload, timeout=60.0)
            if status_code >= 400:
                return f"Error {status_code} fetching documents from Babel: {body}"
            return body
        except Exception as exc:
            return f"Error fetching documents from Babel: {exc}"
