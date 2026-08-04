"""
Heuristic signal estimators feeding the decision policy.

These are deliberately simple keyword/rule heuristics, not models, and are
the pieces of the system most likely to be replaced wholesale by later
phases:
  - Real integrity risk detection belongs with the grading/examiner agent.
    TODO(Phase 4): replace with a real academic-integrity signal.
  - Real mastery estimation requires persisted history across sessions.
    TODO(Phase 5): replace DEFAULT_MASTERY_ESTIMATE / this stub with a
    persisted mastery model.

Both are pure functions over their inputs, kept separate from the decision
policy itself so the policy's unit tests stay about branch logic, not
string matching.
"""

from __future__ import annotations

from app.models.contracts import AssessmentMode, IntegrityRisk

# TODO(Phase 5): source from a persisted per-student mastery model instead
# of a flat neutral prior. Callers that already have a real estimate (e.g.
# a future mastery service) should pass it into handle_turn explicitly.
DEFAULT_MASTERY_ESTIMATE = 0.5

_HIGH_RISK_PHRASES = (
    "give me the full answer",
    "just give me the answer",
    "write the whole solution",
    "do my homework",
    "solve it for me completely",
    "just solve it",
    "give me the solution",
)

_MEDIUM_RISK_PHRASES = (
    "what's the answer",
    "whats the answer",
    "can you just tell me",
    "skip to the answer",
)

_FRUSTRATION_MARKERS = (
    "i don't get it",
    "i dont get it",
    "this is so hard",
    "i give up",
    "i'm so confused",
    "im so confused",
    "ugh",
    "this is stupid",
    "i hate this",
    "so frustrated",
    "i can't do this",
    "i cant do this",
)


def estimate_integrity_risk(raw_input: str, assessment_mode: AssessmentMode) -> IntegrityRisk:
    lowered = raw_input.lower()
    high_risk_request = any(phrase in lowered for phrase in _HIGH_RISK_PHRASES)
    medium_risk_request = any(phrase in lowered for phrase in _MEDIUM_RISK_PHRASES)

    if assessment_mode == AssessmentMode.LIVE_EXAM_SIMULATION:
        # The hard gate on assessment_mode alone (in the decision policy)
        # already refuses live exams; risk is reported as at least medium
        # here for observability/logging even though it isn't what trips
        # the gate.
        return IntegrityRisk.HIGH if high_risk_request else IntegrityRisk.MEDIUM

    if high_risk_request:
        return IntegrityRisk.HIGH if assessment_mode == AssessmentMode.GRADED_TAKE_HOME else IntegrityRisk.MEDIUM

    if medium_risk_request:
        return IntegrityRisk.MEDIUM

    return IntegrityRisk.NONE


def estimate_frustration(raw_input: str) -> bool:
    lowered = raw_input.lower()
    return any(marker in lowered for marker in _FRUSTRATION_MARKERS)
