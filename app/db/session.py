"""
Minimal Postgres skeleton for Phase 0. Nothing in Phase 1 reads or writes
curriculum/persistence data through this — it exists only so the gateway
has a real async engine wired up for later phases to build on.

TODO(Phase 2+): actual tables (curriculum content, retrieval indexes, CAS
verification logs) land on top of this engine.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False, future=True)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with async_session_factory() as session:
        yield session
