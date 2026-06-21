"""Legacy ASGI entry point. Prefer ``api.app:app``."""

from api.app import app

__all__ = ["app"]
