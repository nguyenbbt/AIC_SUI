"""ASGI entrypoint for the AIC Online retrieval API."""

from .search_engine import create_app


app = create_app()


__all__ = ["app", "create_app"]
