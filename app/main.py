"""
FastAPI gateway. MVP pass: every store handle_turn can use is now backed
by a real SQLite file (app/storage/), so a demo student's mastery,
review schedule, IA project state, and every other piece of state
survives a server restart — the previous Phase 0 gateway only persisted
the hint ladder (and only when Redis happened to be running); everything
else reset to empty on every restart. Also serves a minimal static chat
UI at `/` so this is something you can actually open in a browser, not
just curl.

The model layer defaults to Groq (a free-tier provider, see
app/llm/router_config.py) so the whole system runs without a paid API
key — set GROQ_API_KEY and go. Swapping any capability back to Anthropic
(or registering a different provider) is still the one-line
router_config.py change that file's own docstring promises; nothing
here needs to change.
"""

from __future__ import annotations

import base64
import binascii
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.adaptive.store import ReviewStateStore, SqliteReviewStateStore
from app.config import settings
from app.diagnostician.catalog import describe as describe_misconception
from app.guardrail_metrics.store import GuardrailMetricsStore, SqliteGuardrailMetricsStore
from app.ia_supervisor.disclosure_store import DisclosureStore, SqliteDisclosureStore
from app.ia_supervisor.project_store import IAProjectStateStore, SqliteIAProjectStateStore
from app.llm.client import AnthropicProvider, GroqProvider, ModelRouter
from app.llm.router_config import Provider
from app.memory.store import MemoryStore, SqliteMemoryStore
from app.orchestrator.handle_turn import handle_turn
from app.questions.response_log import ResponseLogStore, SqliteResponseLogStore
from app.review_queue.store import ReviewQueueStore, SqliteReviewQueueStore
from app.session.state import SessionStateStore, SqliteSessionStateStore
from app.storage.schema import init_sqlite_schema

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

# Every store below is constructed once at import time, all pointed at
# the same SQLite file (settings.sqlite_db_path) — real, shared,
# persistent state, not a per-request throwaway. Each class ensures its
# own table exists lazily on first use (SqliteBackedStore), and
# app.main's lifespan also calls init_sqlite_schema explicitly below so
# the schema is visibly ready (and logged) before the first request
# rather than paying that cost silently on it.
_session_store: SessionStateStore = SqliteSessionStateStore(settings.sqlite_db_path)
_memory_store: MemoryStore = SqliteMemoryStore(settings.sqlite_db_path)
_review_store: ReviewStateStore = SqliteReviewStateStore(settings.sqlite_db_path)
_ia_project_store: IAProjectStateStore = SqliteIAProjectStateStore(settings.sqlite_db_path)
_ia_disclosure_store: DisclosureStore = SqliteDisclosureStore(settings.sqlite_db_path)
_response_log_store: ResponseLogStore = SqliteResponseLogStore(settings.sqlite_db_path)
_review_queue_store: ReviewQueueStore = SqliteReviewQueueStore(settings.sqlite_db_path)
_guardrail_metrics_store: GuardrailMetricsStore = SqliteGuardrailMetricsStore(settings.sqlite_db_path)
# Explicit providers, constructed from `settings` (which reads GROQ_API_KEY/
# ANTHROPIC_API_KEY from the process environment or a .env file) rather than
# each provider class's own os.environ-only fallback - the .env-file case
# would otherwise silently fail to reach the actual HTTP call even though
# `settings.groq_api_key` looks populated.
_router = ModelRouter(
    providers={
        Provider.ANTHROPIC: AnthropicProvider(api_key=settings.anthropic_api_key),
        Provider.GROQ: GroqProvider(api_key=settings.groq_api_key),
    }
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
    await init_sqlite_schema(settings.sqlite_db_path)
    logger.info("SQLite schema ready at %s", settings.sqlite_db_path)
    if not settings.groq_api_key:
        logger.warning(
            "GROQ_API_KEY is not set - every capability in app/llm/router_config.py "
            "defaults to Groq, so turns will fail with ModelUnavailableError until "
            "it's set (or router_config.py's CAPABILITY_MODEL_MAP is pointed at a "
            "provider that does have a key configured)."
        )
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


class TurnRequest(BaseModel):
    raw_input: str
    session_id: str
    student_id: str
    problem_id: Optional[str] = None
    # Typed working, for a check_work grading — omit for every other intent.
    student_work: Optional[str] = None
    # Base64-encoded PNG/JPEG bytes, the browser alternative to
    # student_work: read via FileReader in the frontend, decoded back to
    # raw bytes here before being handed to the real multimodal ingestion
    # pipeline exactly as student_work_image already expects (Phase 7).
    # Ignored if student_work is also provided (handle_turn's own rule:
    # typed work always wins over a photo for the same turn).
    student_work_image_base64: Optional[str] = None
    responding_to_template_id: Optional[str] = None


class TurnResponse(BaseModel):
    turn_id: str
    text: str
    citations: list[str] = []
    intent: Optional[str] = None
    action_type: Optional[str] = None
    ui_metadata: dict[str, Any] = {}
    # Present only for a real check_work grading (Phase 4).
    mark_result: Optional[dict[str, Any]] = None
    # Present only when a misconception was confidently diagnosed (Phase 8).
    misconception: Optional[dict[str, Any]] = None
    # The generated item's stem only, when CHALLENGE/retrieval-practice
    # served one — never the item's own answer (same structural rule the
    # Tutor agent itself enforces on generated_item, spec §9's leak-check).
    generated_item_stem: Optional[str] = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.post("/turn", response_model=TurnResponse)
async def turn(request: TurnRequest) -> JSONResponse:
    student_work_image: Optional[bytes] = None
    if request.student_work_image_base64 and not request.student_work:
        try:
            student_work_image = base64.b64decode(request.student_work_image_base64, validate=True)
        except (binascii.Error, ValueError):
            return JSONResponse(status_code=400, content={"detail": "student_work_image_base64 is not valid base64"})

    blackboard = await handle_turn(
        raw_input=request.raw_input,
        session_id=request.session_id,
        student_id=request.student_id,
        problem_id=request.problem_id,
        router=_router,
        session_store=_session_store,
        memory_store=_memory_store,
        review_store=_review_store,
        ia_project_store=_ia_project_store,
        ia_disclosure_store=_ia_disclosure_store,
        response_log_store=_response_log_store,
        review_queue_store=_review_queue_store,
        guardrail_metrics_store=_guardrail_metrics_store,
        student_work=request.student_work,
        student_work_image=student_work_image,
        responding_to_template_id=request.responding_to_template_id,
    )

    assert blackboard.final_response is not None
    mark_result = None
    if blackboard.mark_result is not None:
        mr = blackboard.mark_result
        mark_result = {
            "total_awarded": mr.total_awarded,
            "total_available": mr.total_available,
            "confidence": mr.confidence.value,
            "comment": mr.comment,
            "flags": mr.flags,
        }

    misconception = None
    if blackboard.diagnosis_result is not None and blackboard.diagnosis_result.misconception_id is not None:
        misconception = {
            "misconception_id": blackboard.diagnosis_result.misconception_id,
            "description": describe_misconception(blackboard.diagnosis_result.misconception_id),
            "confidence": blackboard.diagnosis_result.confidence,
        }

    response = TurnResponse(
        turn_id=blackboard.turn_id,
        text=blackboard.final_response.text,
        citations=blackboard.final_response.citations,
        intent=blackboard.intent_result.intent.value if blackboard.intent_result else None,
        action_type=blackboard.decision_action.action_type.value if blackboard.decision_action else None,
        ui_metadata=blackboard.final_response.ui_metadata,
        mark_result=mark_result,
        misconception=misconception,
        generated_item_stem=blackboard.generated_item.rendered_stem if blackboard.generated_item else None,
    )
    return JSONResponse(content=response.model_dump(mode="json"))
