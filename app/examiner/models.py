"""
Typed contracts for the Grader / Examiner Agent (spec §10), scoped to what
this phase actually implements: a single-path mark-scheme DAG (reusing
Phase 3's MarkScheme/MarkSchemeNode), real step segmentation, and
symbolic-equivalence alignment for accuracy nodes.

Not modeled here (later-phase non-goals):
  - Alternative solution paths (§10.1's alternative_paths) — one canonical
    CAS path only.
  - Grade boundary / IB 1-7 grade calibration (§10.8) — needs historical
    exam data this system doesn't have.
  - Human-in-the-loop review/appeals queue persistence (§10.10).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class StepType(str, Enum):
    ALGEBRAIC_MANIPULATION = "algebraic_manipulation"
    FINAL_ANSWER = "final_answer"
    JUSTIFICATION = "justification"
    RESTATEMENT_OF_GIVEN = "restatement_of_given"


class WorkStep(BaseModel):
    step_index: int
    raw_text: str
    normalized_expr: Optional[str] = None
    step_type: StepType


class MarkAward(BaseModel):
    node_id: str
    type: str  # "M" or "A", mirrors MarkSchemeNode.type
    marks_available: int
    marks_awarded: int
    matched_step_index: Optional[int] = None
    reason: str


class ConfidenceTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MarkResult(BaseModel):
    item_id: str
    total_awarded: int
    total_available: int
    breakdown: list[MarkAward]
    method_marks: int
    accuracy_marks: int
    first_error_step_index: Optional[int] = None
    flags: list[str] = Field(default_factory=list)
    confidence: ConfidenceTier
    comment: str = ""
