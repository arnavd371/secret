"""
Typed contracts shared across the orchestrator, agents, and policy layer.

Field names and enum values are pinned to the engineering spec (IB Mathematics
AA — AI-Native Academic Assistant, Engineering Blueprint v1.0), specifically:
  - IntentResult:      spec §2.2, Router/Intent Agent
  - DecisionSignals:    spec §1.5, signal table
  - Action:             spec §1.5, decision-policy pseudocode (Action.* calls)
  - TutorResponse:      spec §2.2, Tutor/Teaching Agent output
  - Blackboard:         spec §2.5, Shared Blackboard / State Object Schema

These are the load-bearing data shapes of the system. Every boundary between
components (Router -> Orchestrator -> Policy -> Tutor agent) passes one of
these types, never a raw dict and never a free-form string, so a malformed
value fails validation at the boundary instead of propagating silently into
prompt text or business logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from app.cas.models import CASResult
from app.diagnostician.models import DiagnosisResult
from app.examiner.models import MarkResult
from app.knowledge.schemas import RetrievedChunk
from app.memory.models import MemoryReadContext
from app.multimodal.models import IngestionResult
from app.questions.models import GeneratedItem


# ---------------------------------------------------------------------------
# Shared enums (values pinned to spec §1.5 / §2.2 signal & schema tables)
# ---------------------------------------------------------------------------


class IntentType(str, Enum):
    SOLVE_REQUEST = "solve_request"
    CHECK_WORK = "check_work"
    CONCEPT_EXPLAIN = "concept_explain"
    EXAM_PREP = "exam_prep"
    IA_EE_HELP = "ia_ee_help"
    GENERAL_CHAT = "general_chat"


class AssessmentMode(str, Enum):
    PRACTICE = "practice"
    HOMEWORK_UNGRADED = "homework_ungraded"
    GRADED_TAKE_HOME = "graded_take_home"
    LIVE_EXAM_SIMULATION = "live_exam_simulation"
    IA_EE = "ia_ee"


class IntegrityRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FrustrationLevel(str, Enum):
    NONE = "none"
    MILD = "mild"
    HIGH = "high"


class ActionType(str, Enum):
    EXPLAIN = "explain"
    HINT = "hint"
    QUESTION = "question"
    REFUSE = "refuse"
    CHALLENGE = "challenge"
    SUPPORTIVE_SCAFFOLD = "supportive_scaffold"


MAX_HINT_LADDER_LEVEL = 4


# ---------------------------------------------------------------------------
# Router / Intent agent output (spec §2.2)
# ---------------------------------------------------------------------------


class IntentResult(BaseModel):
    """Structured output of the Router/Intent agent (small/fast model call)."""

    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    subject: str = "math_aa"
    topic_hint: Optional[str] = None
    assessment_mode_guess: AssessmentMode = AssessmentMode.PRACTICE
    requires_multimodal_parse: bool = False
    language: str = "en"


# ---------------------------------------------------------------------------
# Decision policy input/output (spec §1.5)
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
    frustration_signal: FrustrationLevel
    hint_ladder_level: int = Field(ge=0, le=MAX_HINT_LADDER_LEVEL)


class Action(BaseModel):
    """
    The binding contract handed to the Tutor agent. Per spec §1.5: "its
    output (Action) is the binding contract passed to the Tutor agent's
    generation call — the Tutor agent cannot override the action type, only
    the content within it." That structural guarantee is enforced in
    app/agents/tutor_agent.py, not merely requested via prompt wording.

    `offer` may be a single string (e.g. "concept_explanation") or a list
    of alternatives (e.g. ["concept_explanation", "analog_practice_problem"])
    per the spec's REFUSE(offer=...) calls. `tone` / `reduce_difficulty` are
    only meaningful on SUPPORTIVE_SCAFFOLD actions.

    `reason` is an engineering addition beyond the spec (not in the
    Action.* calls) kept for observability/testability — every branch of
    the policy tags why it fired, which costs nothing and makes the pure
    function's behavior auditable in logs and tests.
    """

    action_type: ActionType
    move: Optional[str] = None
    level: Optional[int] = Field(default=None, ge=1, le=MAX_HINT_LADDER_LEVEL)
    offer: Optional[Union[str, list[str]]] = None
    tone: Optional[str] = None
    reduce_difficulty: Optional[bool] = None
    reason: str = Field(min_length=1)

    def model_post_init(self, __context: Any) -> None:
        if self.action_type == ActionType.HINT and self.level is None:
            raise ValueError("HINT actions must carry an explicit level (1-4)")
        if self.action_type != ActionType.HINT and self.level is not None:
            raise ValueError("level is only meaningful on HINT actions")


# ---------------------------------------------------------------------------
# Tutor agent output (spec §2.2)
# ---------------------------------------------------------------------------


class TutorResponse(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)
    ui_metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Safety/Integrity agent output (spec §2.2) — Phase 1 implements this via a
# small keyword heuristic (app/orchestrator/signals.py), not a trained
# classifier. TODO(later phase): replace with the real Safety/Integrity agent.
# ---------------------------------------------------------------------------


class SafetyResult(BaseModel):
    integrity_risk: IntegrityRisk
    pii_flags: list[str] = Field(default_factory=list)
    content_safety_flags: list[str] = Field(default_factory=list)
    action_required: bool = False


# ---------------------------------------------------------------------------
# Blackboard: shared per-turn state object (spec §2.5)
# ---------------------------------------------------------------------------


class RawInput(BaseModel):
    text: str
    attachments: list[str] = Field(default_factory=list)


class NormalizedInput(BaseModel):
    text: str
    # TODO(Phase 2/7): populated by the multimodal/CAS-expression pipeline.
    expression_trees: list[str] = Field(default_factory=list)
    # TODO(Phase 7): populated by the math-OCR pipeline (§3.2) for image input.
    ocr_confidence: Optional[float] = None


class Blackboard(BaseModel):
    """
    Shared state object threaded through a single turn's processing,
    matching the field names of spec §2.5 exactly so later phases can
    populate the fields Phase 1 leaves as stubs without a rename.
    """

    turn_id: str
    session_id: str
    student_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    raw_input: RawInput
    normalized_input: Optional[NormalizedInput] = None

    intent_result: Optional[IntentResult] = None
    safety_result: Optional[SafetyResult] = None

    # Populated by the Memory agent (app/memory/context_assembly.py) when
    # the turn's topic_hint resolves to a persisted mastery record — the
    # spec's own "LearnerModel summary, §4.2" field, now the real
    # deterministic budgeted-context-assembly output of §4.12 rather than
    # a stub. None when no topic_hint was available or no record exists
    # yet for this student/subtopic.
    student_state_snapshot: Optional[MemoryReadContext] = None
    # TODO: Planner agent isn't built in Phase 1 — the orchestrator runs a
    # fixed linear sequence instead of a Planner-produced stage graph.
    execution_plan: Optional[dict[str, Any]] = None
    # Populated by the Retriever agent (app/knowledge/retriever.py) when the
    # decision_action is EXPLAIN. TODO(later phase): the real hybrid
    # BM25+dense+graph+rerank pipeline of §5.6 — this is a lexical
    # TF-IDF/cosine retriever over a small hand-authored seed corpus, not
    # the full knowledge base.
    retrieved_chunks: Optional[list[RetrievedChunk]] = None
    # Populated by the Math Solver + CAS agent (app/cas/solver.py) when a
    # math task is extractable from the turn and the decision_action is
    # EXPLAIN. None when no task was extracted this turn.
    cas_result: Optional[CASResult] = None
    # Populated by the Misconception Diagnostician (app/diagnostician/diagnose.py,
    # spec §8) after a check_work submission grades as incorrect: a real
    # pattern-matched or model-inferred diagnosis, or a "no diagnosis"
    # result when nothing was confidently identifiable. None when the
    # turn wasn't a graded-incorrect check_work submission at all.
    diagnosis_result: Optional[DiagnosisResult] = None
    # Populated by the Question Generation Engine (app/questions/generator.py)
    # when the decision_action is CHALLENGE — a real, CAS-verified,
    # quality-gated extension item, not something the Tutor agent invents.
    # Not a field named in spec §2.5's Blackboard schema directly; added
    # here as the natural home for it, same spirit as the schema's other
    # per-agent-output fields.
    generated_item: Optional[GeneratedItem] = None
    # Populated by the Grader/Examiner agent (app/examiner/grader.py) when
    # the turn is a check_work submission accompanied by student_work text
    # and the problem it's checking against was extractable via CAS. Not a
    # field named in spec §2.5's Blackboard schema directly, same rationale
    # as generated_item above.
    mark_result: Optional[MarkResult] = None
    # Populated by the multimodal ingestion pipeline (app/multimodal/pipeline.py,
    # spec §3.2) when the turn arrives with a `student_work_image` — real
    # intake validation, PIL preprocessing, vision-model OCR, LaTeX
    # normalization, expression-parseability check, and composite
    # confidence scoring. None when the turn had no image. Same rationale
    # as generated_item/mark_result above: not a field named in spec
    # §2.5's Blackboard schema directly, added as the natural home for it.
    ingestion_result: Optional[IngestionResult] = None

    decision_action: Optional[Action] = None
    draft_response: Optional[TutorResponse] = None
    # TODO: full Verifier/Critic agent is a later phase; Phase 1 approximates
    # its no-leak check with a structural regex gate inside tutor_agent.py.
    critique_result: Optional[dict[str, Any]] = None
    final_response: Optional[TutorResponse] = None

    # TODO(Phase 5): async memory write-back queue.
    memory_writes_pending: list[dict[str, Any]] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
