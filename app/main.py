"""
Phase 0 FastAPI gateway, kept minimal — a health check and a single /turn
endpoint that streams a Tutor response by delegating to the Phase 1
orchestrator. Redis is wired up when available and the app transparently
falls back to the in-memory session store otherwise, so local dev without
Redis running still works.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.orchestrator.handle_turn import handle_turn
from app.session.state import InMemorySessionStateStore, RedisSessionStateStore, SessionStateStore

logger = logging.getLogger(__name__)

_session_store: SessionStateStore = InMemorySessionStateStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _session_store
    try:
        import redis.asyncio as redis

        client = redis.from_url(settings.redis_url, decode_responses=True)
        await client.ping()
        _session_store = RedisSessionStateStore(client)
        logger.info("Connected to Redis at %s", settings.redis_url)
    except Exception as exc:  # noqa: BLE001 - Redis is optional in dev
        logger.warning("Redis unavailable (%s); using in-memory session store", exc)
        _session_store = InMemorySessionStateStore()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


class TurnRequest(BaseModel):
    raw_input: str
    session_id: str
    student_id: str
    problem_id: str | None = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/turn")
async def turn(request: TurnRequest) -> StreamingResponse:
    blackboard = await handle_turn(
        raw_input=request.raw_input,
        session_id=request.session_id,
        student_id=request.student_id,
        problem_id=request.problem_id,
        session_store=_session_store,
    )

    async def _stream():
        assert blackboard.tutor_response is not None
        for chunk in [
            blackboard.tutor_response.text[i : i + 40]
            for i in range(0, len(blackboard.tutor_response.text), 40)
        ]:
            yield chunk

    return StreamingResponse(_stream(), media_type="text/plain")
