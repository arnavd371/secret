"""
Tests for the Vercel deployment adapter: the ASGI entrypoint actually
re-exports a working app, and the SQLite path really does switch to
Vercel's writable /tmp when the platform's own VERCEL env var is set -
the one runtime difference that matters between "runs locally" and
"runs on Vercel's read-only-except-/tmp filesystem".
"""

from __future__ import annotations

import importlib

import httpx
import pytest


def test_api_entrypoint_reexports_the_real_fastapi_app():
    from api.index import app
    from app.main import app as main_app

    assert app is main_app
    assert app.title == "ai-tutor-gateway"


@pytest.mark.asyncio
async def test_api_entrypoint_serves_real_requests():
    from api.index import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_sqlite_path_defaults_to_tmp_under_vercel(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    import app.config as config_module

    importlib.reload(config_module)
    try:
        assert config_module.Settings().sqlite_db_path == "/tmp/tutor.db"
    finally:
        monkeypatch.delenv("VERCEL", raising=False)
        importlib.reload(config_module)


def test_sqlite_path_defaults_to_local_data_dir_outside_vercel(monkeypatch):
    monkeypatch.delenv("VERCEL", raising=False)
    import app.config as config_module

    importlib.reload(config_module)
    assert config_module.Settings().sqlite_db_path == "data/tutor.db"


def test_sqlite_path_is_still_overridable_via_env_var(monkeypatch):
    monkeypatch.setenv("SQLITE_DB_PATH", "/custom/path.db")
    import app.config as config_module

    importlib.reload(config_module)
    try:
        assert config_module.Settings().sqlite_db_path == "/custom/path.db"
    finally:
        monkeypatch.delenv("SQLITE_DB_PATH", raising=False)
        importlib.reload(config_module)
