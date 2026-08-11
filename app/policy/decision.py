"""
Deterministic pedagogical decision policy — a direct implementation of the
spec's §1.5 pseudocode for `decide_pedagogical_action`.

This is a pure function: given a DecisionSignals struct, it returns an
Action. No I/O, no model calls, no randomness, no mutation of its input.
That is what makes it exhaustively unit-testable and what makes it
impossible for the LLM to talk its way around a hard gate — the gate is
evaluated in Python before any generation happens.

Branch order matters and is enforced by early return, matching the spec
exactly:
    1. Hard gates (integrity / assessment mode) — always win, no exceptions.
    2. IA/EE routing (never full ghostwriting).
    3. Frustration override.
    4. Mastery-based shortcut for high performers.
    5. Core Socratic ladder for solve_request.
    6. Fallback branches for check_work / concept_explain / exam_prep.
    7. Default fallback for anything else (general_chat).
"""

from __future__ import annotations

from app.models.contracts import (
    Action,
    ActionType,
    AssessmentMode,
    DecisionSignals,
    FrustrationLevel,
    IntegrityRisk,
    IntentType,
    MAX_HINT_LADDER_LEVEL,
)

# Named per spec §1.5's decision table row: "solve_request, practice, low,
# >=0.85, any, none -> brief confirm + challenge/extension question".
HIGH_MASTERY_THRESHOLD = 0.85

# Per spec pseudocode: attempt_count in (2,3) hints are capped at level 2;
# attempt_count >= 4 hints are capped at the ladder's max level.
MID_ATTEMPT_HINT_CAP = 2


def _route_to_ia_supervisor(signals: DecisionSignals) -> Action:
    """
    Spec §1.5: `if signals.intent == "ia_ee_help": return
    route_to_ia_supervisor(signals) # see 2.10 — never full ghostwriting`.

    The full IA Supervisor Agent (spec §11) — state machine, guard
    architecture, disclosure logging — is far outside Phase 1's reasoning
    core and is not built here. This stub preserves the one property that
    actually matters at this layer: IA/EE help is never routed to a full
    solve/explain path. It refuses and offers the same legitimate
    substitute the real IA Supervisor would (methodology/structure
    coaching), rather than silently falling through to a normal EXPLAIN.
    TODO(Section 11): replace with a real call into the IA Supervisor Agent.
    """
    return Action(
        action_type=ActionType.REFUSE,
        offer="ia_methodology_coaching",
        reason="ia_ee_help_routed_to_stub_supervisor",
    )


def _schedule_next_review_item(signals: DecisionSignals) -> Action:
    """
    Spec §1.5: `if signals.intent == "exam_prep": return
    schedule_next_review_item(signals) # spaced repetition, 4.x`, and the
    decision table's own description of this cell: "question
    (retrieval-practice style) or explain, chosen by spaced-review
    scheduler."

    `signals.has_due_review` is the real signal, resolved by the
    orchestrator from the Adaptive Learning Engine's FSRS-based due-review
    queue (app/adaptive/scheduler.py, spec §12) before this pure function
    ever runs. When something is genuinely due, retrieval practice is the
    right move — actively recalling a decaying item is exactly what spaced
    repetition is for. When nothing is due (nothing has entered the
    review cycle yet, or everything is still comfortably within its
    interval), quizzing the student on a term serves no one — a general
    explanation/orientation response is the honest fallback instead.
    """
    if signals.has_due_review:
        return Action(action_type=ActionType.QUESTION, move="retrieval_practice", reason="exam_prep_review_due")
    return Action(action_type=ActionType.EXPLAIN, move="general_response", reason="exam_prep_no_review_due")


def decide_pedagogical_action(signals: DecisionSignals) -> Action:
    # ------------------------------------------------------------------
    # 1. Hard gates. These short-circuit before anything else runs, full
    #    stop. No branch below this block may override a gate.
    # ------------------------------------------------------------------
    if signals.integrity_risk == IntegrityRisk.HIGH:
        return Action(action_type=ActionType.REFUSE, offer="concept_explanation", reason="integrity_risk_high")

    if signals.assessment_mode == AssessmentMode.LIVE_EXAM_SIMULATION:
        return Action(
            action_type=ActionType.REFUSE, offer="strategy_coaching_only", reason="live_exam_simulation"
        )

    if signals.assessment_mode == AssessmentMode.GRADED_TAKE_HOME and signals.integrity_risk in (
        IntegrityRisk.MEDIUM,
        IntegrityRisk.HIGH,
    ):
        return Action(
            action_type=ActionType.REFUSE,
            offer=["concept_explanation", "analog_practice_problem"],
            reason="graded_take_home_elevated_risk",
        )

    # ------------------------------------------------------------------
    # 2. IA/EE routing — never full ghostwriting, regardless of other
    #    signals (this check runs before frustration/mastery so an IA
    #    request never accidentally gets CHALLENGE'd or scaffolded into
    #    a normal solve path).
    # ------------------------------------------------------------------
    if signals.intent == IntentType.IA_EE_HELP:
        return _route_to_ia_supervisor(signals)

    # ------------------------------------------------------------------
    # 3. Frustration override: de-escalate hint ladder, protect
    #    engagement. Only the "high" tier overrides — "mild" does not.
    # ------------------------------------------------------------------
    if signals.frustration_signal == FrustrationLevel.HIGH:
        return Action(
            action_type=ActionType.SUPPORTIVE_SCAFFOLD,
            move="worked_example_with_fade",
            tone="reassuring",
            reduce_difficulty=True,
            reason="frustration_override",
        )

    # ------------------------------------------------------------------
    # 4. Mastery-based shortcut: high performers get challenge, not
    #    repetition. Per spec pseudocode this fires only on the FIRST
    #    attempt at a solve_request (attempt_count == 1) — a student who
    #    is on their third attempt is not a "high performer on this
    #    problem" even if their historical mastery estimate is high.
    # ------------------------------------------------------------------
    if (
        signals.intent == IntentType.SOLVE_REQUEST
        and signals.mastery_estimate >= HIGH_MASTERY_THRESHOLD
        and signals.attempt_count == 1
    ):
        return Action(action_type=ActionType.CHALLENGE, move="extension_question", reason="high_mastery_shortcut")

    # ------------------------------------------------------------------
    # 5. Core Socratic ladder for solve requests.
    #
    #    attempt_count <= 1        -> diagnostic Socratic question.
    #    attempt_count in (2, 3)   -> hint, escalating by one level each
    #                                 time, capped at level 2.
    #    attempt_count >= 4        -> hint, escalating by one level each
    #                                 time, capped at the ladder max (4).
    # ------------------------------------------------------------------
    if signals.intent == IntentType.SOLVE_REQUEST:
        if signals.attempt_count <= 1:
            return Action(action_type=ActionType.QUESTION, move="diagnostic_probe", reason="first_attempt_socratic")
        if signals.attempt_count in (2, 3):
            level = min(signals.hint_ladder_level + 1, MID_ATTEMPT_HINT_CAP)
            return Action(action_type=ActionType.HINT, level=level, reason=f"solve_request_attempt_{signals.attempt_count}")
        level = min(signals.hint_ladder_level + 1, MAX_HINT_LADDER_LEVEL)
        return Action(action_type=ActionType.HINT, level=level, reason="solve_request_attempt_4_plus")

    # ------------------------------------------------------------------
    # 6. Fallback branches for non-ladder intents.
    # ------------------------------------------------------------------
    if signals.intent == IntentType.CHECK_WORK:
        return Action(action_type=ActionType.EXPLAIN, move="error_localization_explanation", reason="check_work")

    if signals.intent == IntentType.CONCEPT_EXPLAIN:
        return Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="concept_explain")

    if signals.intent == IntentType.EXAM_PREP:
        return _schedule_next_review_item(signals)

    # ------------------------------------------------------------------
    # 7. Default fallback (general_chat, or any future intent value not
    #    explicitly handled above).
    # ------------------------------------------------------------------
    return Action(action_type=ActionType.EXPLAIN, move="general_response", reason="default_fallback")
