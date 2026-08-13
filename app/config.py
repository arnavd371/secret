"""
Minimal Phase 0 settings. Kept intentionally small — Phase 0's FastAPI +
Postgres + Redis skeleton is assumed/scaffolded here only as far as Phase 1
needs it to run; it is not itself the focus of this phase.
"""

from __future__ import annotations

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_sqlite_path() -> str:
    """Vercel's serverless filesystem is read-only outside /tmp, and
    /tmp itself is wiped between cold starts and isn't shared across
    concurrent instances - so this is NOT durable persistence the way
    "data/tutor.db" is for a normal long-running local/server process.
    Vercel sets its own VERCEL env var at runtime, which is the real
    signal to use here rather than guessing from environment='dev'."""
    return "/tmp/tutor.db" if os.environ.get("VERCEL") else "data/tutor.db"


class Settings(BaseSettings):
    app_name: str = "ai-tutor-gateway"
    environment: str = "dev"

    database_url: str = "postgresql+asyncpg://tutor:tutor@localhost:5432/tutor"
    redis_url: str = "redis://localhost:6379/0"

    # MVP persistence: a single SQLite file backs every store (see
    # app/storage/schema.py) - no Postgres/Redis server needed to run
    # this locally with state that survives a restart. database_url and
    # redis_url above remain unused by app.main for now, kept as the
    # documented path to a real multi-instance deployment later.
    sqlite_db_path: str = Field(default_factory=_default_sqlite_path)

    anthropic_api_key: str | None = None
    groq_api_key: str | None = None

    tutor_generate_timeout_seconds: float = 8.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
