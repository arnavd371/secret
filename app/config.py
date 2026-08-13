"""
Minimal Phase 0 settings. Kept intentionally small — Phase 0's FastAPI +
Postgres + Redis skeleton is assumed/scaffolded here only as far as Phase 1
needs it to run; it is not itself the focus of this phase.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    sqlite_db_path: str = "data/tutor.db"

    anthropic_api_key: str | None = None
    groq_api_key: str | None = None

    tutor_generate_timeout_seconds: float = 8.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
