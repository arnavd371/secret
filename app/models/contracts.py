"""
Typed contracts shared across the orchestrator, agents, and policy layer.

These are the load-bearing data shapes of the system. Every boundary between
components (Router -> Orchestrator -> Policy -> Tutor agent) passes one of
these types, never a raw dict and never a free-form string, so a malformed
value fails validation at the boundary instead of propagating silently into
prompt text or business logic.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------


class IntentType(str, Enum):
    CONCEPT_EXPLAIN = "concept_explain"
    PRACTICE = "practice"
    CHECK_WORK = "check_work"
    EXAM_PREP = "exam_prep"
    HINT_REQUEST = "hint_request"
    OFF_TOPIC = "off_topic"


class AssessmentMode(str, Enum):
    PRACTICE = "practice"
    GRADED_TAKE_HOME = "graded_take_home"
    LIVE_EXAM_SIMULATION = "live_exam_simulation"
    NONE = "none"


class IntegrityRisk(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionType(str, Enum):
    EXPLAIN = "explain"
    HINT = "hint"
    QUESTION = "question"
    REFUSE = "refuse"
    CHALLENGE = "challenge"
    SUPPORTIVE_SCAFFOLD = "supportive_scaffold"


# ---------------------------------------------------------------------------
# Router / Intent agent output
# ---------------------------------------------------------------------------


class IntentResult(BaseModel):
    """Structured output of the Router/Intent agent (small/fast model call)."""

    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    subject: str
    topic_hint: Optional[str] = None
    assessment_mode_guess: AssessmentMode = AssessmentMode.NONE
    requires_multimodal_parse: bool = False
    language: str = "en"


# ---------------------------------------------------------------------------
# Decision policy input/output
# ---------------------------------------------------------------------------


class DecisionSignals(BaseModel):
    """
    Everything the pure decision policy needs to pick an Action.

    This struct is intentionally flat and fully deterministic-input: no
    object references, no callables, nothing that could make the policy
    function impure by accident.
    """

    intent: IntentType
    mastery_estimate: float = Field(ge=0.0, le=1.0)
    assessment_mode: AssessmentMode
    integrity_risk: IntegrityRisk
    attempt_count: int = Field(ge=0)
    frustration_signal: bool
    hint_ladder_level: int = Field(ge=0, le=4)


class Action(BaseModel):
    """
    The binding contract handed to the Tutor agent. The agent MUST NOT
    deviate from action_type/level — that is enforced structurally in
    app/agents/tutor_agent.py, not merely requested via prompt wording.
    """

    action_type: ActionType
    move: Optional[str] = None
    level: Optional[int] = Field(default=None, ge=0, le=4)
    offer: Optional[str] = None
    reason: str = Field(min_length=1)

    def model_post_init(self, __context: Any) -> None:
        if self.action_type == ActionType.HINT and self.level is None:
            raise ValueError("HINT actions must carry an explicit level (0-4)")


# ---------------------------------------------------------------------------
# Tutor agent output
# ---------------------------------------------------------------------------


class TutorResponse(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)
    ui_metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Blackboard: shared per-turn state object
# ---------------------------------------------------------------------------


class Blackboard(BaseModel):
    """
    Shared state object threaded through a single turn's processing.

    Only Phase 1-relevant fields are populated. Fields owned by later
    phases are declared as typed-but-unused stubs so downstream code can
    already reference the eventual shape without rework.
    """

    session_id: str
    student_id: str
    problem_id: Optional[str] = None
    raw_input: str

    intent_result: Optional[IntentResult] = None
    decision_signals: Optional[DecisionSignals] = None
    action: Optional[Action] = None
    tutor_response: Optional[TutorResponse] = None

    # TODO(Phase 2): populated by the retrieval agent against the curriculum
    # knowledge base. Left as None/empty in Phase 1 — no retrieval exists yet.
    retrieved_chunks: Optional[list[Any]] = None

    # TODO(Phase 2): populated by the CAS/SymPy verification service when a
    # student submits a symbolic/numeric answer for correctness checking.
    cas_result: Optional[dict[str, Any]] = None
