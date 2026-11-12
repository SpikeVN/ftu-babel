"""FastAPI Application factory and instance for Babel."""

from __future__ import annotations

from fastapi import FastAPI

from server.routes import docs_router, motd_router, search_router


def create_app() -> FastAPI:
    """Create and configure the Babel FastAPI application."""
    application = FastAPI(
        title="FTULabs Babel",
        description="Hybrid-searchable knowledge base system",
        version="1.0.0",
    )

    # Register routers
    application.include_router(search_router)
    application.include_router(docs_router)
    application.include_router(motd_router)

    return application


app = create_app()
