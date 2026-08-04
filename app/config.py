"""
Minimal Phase 0 settings. Kept intentionally small — Phase 0's FastAPI +
Postgres + Redis skeleton is assumed/scaffolded here only as far as Phase 1
needs it to run; it is not itself the focus of this phase.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "ai-tutor-gateway"
    environment: str = "dev"

    database_url: str = "postgresql+asyncpg://tutor:tutor@localhost:5432/tutor"
    redis_url: str = "redis://localhost:6379/0"

    anthropic_api_key: str | None = None

    tutor_generate_timeout_seconds: float = 8.0

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
