"""Babel server routes package."""

from server.routes.docs import router as docs_router
from server.routes.motd import router as motd_router
from server.routes.search import router as search_router

__all__ = ["search_router", "docs_router", "motd_router"]
