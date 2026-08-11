"""
Heuristic signal estimators feeding the decision policy.

Per spec §2.2, integrity_risk should come from a dedicated Safety/Integrity
Agent (fast classifier + rule/keyword lists) and frustration_signal from a
"Tutor agent sentiment/behavior model" tracking repeated negative replies,
rapid re-submission, and explicit "I don't get it" x N. Phase 1 implements
both as small, explicit keyword/rule heuristics rather than trained
classifiers — deterministic and testable, but a stand-in for the real
agents. TODO(later phase): replace with the real Safety/Integrity Agent and
a real frustration-detection model.

Both are pure functions over their inputs, kept separate from the decision
policy itself so the policy's unit tests stay about branch logic, not
string matching.
"""

from __future__ import annotations

from app.memory.bkt import P_INIT_DEFAULT
from app.models.contracts import AssessmentMode, FrustrationLevel, IntegrityRisk, SafetyResult

# Fallback prior used only when the Phase 5 Memory Agent has no persisted
# mastery record yet for the turn's subtopic (a brand-new student, or a
# turn with no resolvable topic_hint at all) — the real per-subtopic BKT
# p_init default (spec §4.3), not the flat 0.5 "neutral prior" this
# constant held before the Memory Agent existed to source a real value.
DEFAULT_MASTERY_ESTIMATE = P_INIT_DEFAULT

_HIGH_RISK_PHRASES = (
    "give me the full answer",
    "just give me the answer",
    "write the whole solution",
    "do my homework",
    "solve it for me completely",
    "just solve it",
    "give me the solution",
    "don't tell my teacher",
    "dont tell my teacher",
)

_MEDIUM_RISK_PHRASES = (
    "what's the answer",
    "whats the answer",
    "can you just tell me",
    "skip to the answer",
    "due tomorrow",
)

_HIGH_FRUSTRATION_MARKERS = (
    "i give up",
    "this is stupid",
    "i hate this",
    "i can't do this",
    "i cant do this",
    "forget it",
    "whatever, i don't care",
)

_MILD_FRUSTRATION_MARKERS = (
    "i don't get it",
    "i dont get it",
    "this is so hard",
    "i'm so confused",
    "im so confused",
    "ugh",
    "so frustrated",
    "still don't get it",
    "still dont get it",
)


def estimate_safety_result(raw_input: str, assessment_mode: AssessmentMode) -> SafetyResult:
    lowered = raw_input.lower()
    high_risk_request = any(phrase in lowered for phrase in _HIGH_RISK_PHRASES)
    medium_risk_request = any(phrase in lowered for phrase in _MEDIUM_RISK_PHRASES)

    if assessment_mode == AssessmentMode.GRADED_TAKE_HOME and high_risk_request:
        risk = IntegrityRisk.HIGH
    elif high_risk_request:
        risk = IntegrityRisk.MEDIUM
    elif medium_risk_request:
        risk = IntegrityRisk.MEDIUM
    else:
        risk = IntegrityRisk.LOW

    return SafetyResult(integrity_risk=risk, action_required=risk == IntegrityRisk.HIGH)


def estimate_frustration(raw_input: str) -> FrustrationLevel:
    lowered = raw_input.lower()
    if any(marker in lowered for marker in _HIGH_FRUSTRATION_MARKERS):
        return FrustrationLevel.HIGH
    if any(marker in lowered for marker in _MILD_FRUSTRATION_MARKERS):
        return FrustrationLevel.MILD
    return FrustrationLevel.NONE
