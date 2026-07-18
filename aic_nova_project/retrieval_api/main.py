"""ASGI entrypoint for the AIC Online retrieval API."""

from .composition import create_runtime_app_from_env
from .search_engine import create_app


app = create_runtime_app_from_env()


__all__ = ["app", "create_app"]
