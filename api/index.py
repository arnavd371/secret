"""
Vercel's Python runtime (@vercel/python) auto-detects an ASGI `app`
object exported from a file under /api and wraps it as a serverless
function - this is the entire adapter needed, no separate WSGI shim or
route re-declaration. Everything real (routes, lifespan, stores, the
model router) still lives in app/main.py; this file exists only because
Vercel's convention requires the entrypoint to sit under /api.
"""

from app.main import app

__all__ = ["app"]
