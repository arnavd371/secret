"""
Real per-student GDPR export/erasure workflows, spanning every store in
this codebase that persists student_id-keyed data: Phase 5's memory
(mastery + misconceptions), Phase 9's adaptive FSRS review state, Phase
10's IA/EE project state and disclosure log, Phase 14's per-item
response history, and Phase 17's review queue.

Export (`export_student_data`) is comprehensive - every store, including
the disclosure log. Erasure (`erase_student_data`) deliberately excludes
the disclosure log: app.ia_supervisor.disclosure_store's own docstring
already establishes those records as append-only and never edited or
deleted, and an AI-use academic-integrity disclosure log is exactly the
kind of record GDPR Article 17(3)'s legal-obligation exemption covers -
a school may need to retain it regardless of an erasure request. This
is a documented, deliberate exemption, not an oversight or a silent gap:
`ErasureReport.disclosure_log_retained` says so explicitly, and the
export still includes it, since an access request and an erasure
request are legally different rights.

Session state (app.session.state.SessionStateStore) is excluded from
both: `ProblemSessionState` is keyed by session_id alone and never
stores student_id, so there's no real query path from "this student" to
"these sessions" without inventing a lookup table this system doesn't
have. Documented here rather than silently ignored.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.adaptive.store import ReviewStateStore
from app.ia_supervisor.disclosure_store import DisclosureStore
from app.ia_supervisor.project_store import IAProjectStateStore
from app.memory.store import MemoryStore
from app.questions.response_log import ResponseLogStore
from app.review_queue.store import ReviewQueueStore


class StudentDataExport(BaseModel):
    student_id: str
    mastery: list[dict[str, Any]] = Field(default_factory=list)
    misconceptions: list[dict[str, Any]] = Field(default_factory=list)
    review_states: list[dict[str, Any]] = Field(default_factory=list)
    ia_projects: list[dict[str, Any]] = Field(default_factory=list)
    disclosure_entries: list[dict[str, Any]] = Field(default_factory=list)
    review_queue_entries: list[dict[str, Any]] = Field(default_factory=list)
    item_responses: list[dict[str, Any]] = Field(default_factory=list)


class ErasureReport(BaseModel):
    student_id: str
    mastery_and_misconceptions_erased: int = 0
    review_states_erased: int = 0
    ia_projects_erased: int = 0
    review_queue_entries_erased: int = 0
    item_responses_erased: int = 0
    # Always True: documents the real, deliberate retention exemption
    # described in this module's docstring rather than leaving erasure
    # scope ambiguous to whoever calls this later.
    disclosure_log_retained: bool = True

    @property
    def total_records_erased(self) -> int:
        return (
            self.mastery_and_misconceptions_erased
            + self.review_states_erased
            + self.ia_projects_erased
            + self.review_queue_entries_erased
            + self.item_responses_erased
        )


async def export_student_data(
    student_id: str,
    *,
    memory_store: MemoryStore,
    review_store: ReviewStateStore,
    ia_project_store: IAProjectStateStore,
    disclosure_store: DisclosureStore,
    review_queue_store: ReviewQueueStore,
    response_log_store: ResponseLogStore,
) -> StudentDataExport:
    mastery = await memory_store.get_all_mastery(student_id)
    misconceptions = await memory_store.get_misconceptions(student_id)
    review_states = await review_store.get_all_for_student(student_id)
    ia_projects = await ia_project_store.get_all_for_student(student_id)
    disclosure_entries = await disclosure_store.get_all_for_student(student_id)
    review_queue_entries = await review_queue_store.get_all_for_student(student_id)
    item_responses = await response_log_store.get_all_for_student(student_id)

    return StudentDataExport(
        student_id=student_id,
        mastery=[m.model_dump(mode="json") for m in mastery],
        misconceptions=[m.model_dump(mode="json") for m in misconceptions],
        review_states=[r.model_dump(mode="json") for r in review_states],
        ia_projects=[p.model_dump(mode="json") for p in ia_projects],
        disclosure_entries=[d.model_dump(mode="json") for d in disclosure_entries],
        review_queue_entries=[r.model_dump(mode="json") for r in review_queue_entries],
        item_responses=[r.model_dump(mode="json") for r in item_responses],
    )


async def erase_student_data(
    student_id: str,
    *,
    memory_store: MemoryStore,
    review_store: ReviewStateStore,
    ia_project_store: IAProjectStateStore,
    review_queue_store: ReviewQueueStore,
    response_log_store: ResponseLogStore,
) -> ErasureReport:
    """No `disclosure_store` parameter, deliberately - see this module's
    docstring for why the disclosure log is exempt from erasure."""
    mastery_count = await memory_store.erase_student(student_id)
    review_count = await review_store.erase_student(student_id)
    ia_count = await ia_project_store.erase_student(student_id)
    review_queue_count = await review_queue_store.erase_student(student_id)
    response_count = await response_log_store.erase_student(student_id)

    return ErasureReport(
        student_id=student_id,
        mastery_and_misconceptions_erased=mastery_count,
        review_states_erased=review_count,
        ia_projects_erased=ia_count,
        review_queue_entries_erased=review_queue_count,
        item_responses_erased=response_count,
    )
