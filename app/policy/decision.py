"""
Deterministic pedagogical decision policy.

`decide_pedagogical_action` is a pure function: given a DecisionSignals
struct, it returns an Action. No I/O, no model calls, no randomness, no
mutation of its input. This is what makes it exhaustively unit-testable and
what makes it impossible for the LLM to talk its way around a hard gate —
the gate is evaluated in Python before any generation happens.

Branch order matters and is enforced by early return:
    1. Hard gates (integrity / exam-mode) — always win, no exceptions.
    2. Frustration override.
    3. Mastery-based shortcut for high performers.
    4. Core Socratic hint ladder (practice / hint_request intents).
    5. Fallback branches for check_work / concept_explain / exam_prep.
    6. Default fallback for anything else (e.g. off_topic).
"""

from __future__ import annotations

from app.models.contracts import (
    Action,
    ActionType,
    AssessmentMode,
    DecisionSignals,
    IntegrityRisk,
    IntentType,
)

# Named thresholds so tests (and future tuning) don't deal in magic numbers.
HIGH_MASTERY_THRESHOLD = 0.85
MAX_HINT_LEVEL = 4

_LADDER_INTENTS = (IntentType.PRACTICE, IntentType.HINT_REQUEST)
_MASTERY_SHORTCUT_INTENTS = (IntentType.PRACTICE, IntentType.EXAM_PREP)


def decide_pedagogical_action(signals: DecisionSignals) -> Action:
    # ------------------------------------------------------------------
    # 1. Hard gates. These short-circuit before anything else runs, full
    #    stop. No branch below this block may override a gate.
    # ------------------------------------------------------------------
    if signals.integrity_risk == IntegrityRisk.HIGH:
        return Action(action_type=ActionType.REFUSE, reason="integrity_risk_high")

    if signals.assessment_mode == AssessmentMode.LIVE_EXAM_SIMULATION:
        return Action(action_type=ActionType.REFUSE, reason="live_exam_simulation")

    if signals.assessment_mode == AssessmentMode.GRADED_TAKE_HOME and signals.integrity_risk in (
        IntegrityRisk.MEDIUM,
        IntegrityRisk.HIGH,
    ):
        return Action(action_type=ActionType.REFUSE, reason="graded_take_home_elevated_risk")

    # ------------------------------------------------------------------
    # 2. Frustration override. Evaluated after gates (a frustrated
    #    student mid-exam-integrity-violation still gets refused), but
    #    before mastery/ladder logic (a frustrated high performer gets
    #    supported, not challenged).
    # ------------------------------------------------------------------
    if signals.frustration_signal:
        return Action(action_type=ActionType.SUPPORTIVE_SCAFFOLD, reason="frustration_override")

    # ------------------------------------------------------------------
    # 3. Mastery-based shortcut for high performers.
    # ------------------------------------------------------------------
    if (
        signals.mastery_estimate >= HIGH_MASTERY_THRESHOLD
        and signals.intent in _MASTERY_SHORTCUT_INTENTS
    ):
        return Action(action_type=ActionType.CHALLENGE, reason="high_mastery_shortcut")

    # ------------------------------------------------------------------
    # 4. Core Socratic hint ladder.
    #
    #    attempt_count == 0            -> always start Socratic (a fresh
    #                                      attempt, regardless of any
    #                                      stale ladder level).
    #    hint_ladder_level == 0        -> keep asking Socratic questions.
    #    0 < hint_ladder_level < MAX   -> hand out the hint at that level.
    #    hint_ladder_level == MAX      -> final hint + offer full
    #                                      solution rather than looping
    #                                      forever.
    # ------------------------------------------------------------------
    if signals.intent in _LADDER_INTENTS:
        if signals.attempt_count == 0:
            return Action(
                action_type=ActionType.QUESTION,
                move="socratic_prompt",
                reason="first_attempt_socratic",
            )

        if signals.hint_ladder_level == 0:
            return Action(
                action_type=ActionType.QUESTION,
                move="socratic_prompt",
                reason="no_hint_yet_continue_socratic",
            )

        if signals.hint_ladder_level < MAX_HINT_LEVEL:
            return Action(
                action_type=ActionType.HINT,
                level=signals.hint_ladder_level,
                reason=f"hint_ladder_level_{signals.hint_ladder_level}",
            )

        return Action(
            action_type=ActionType.HINT,
            level=MAX_HINT_LEVEL,
            offer="offer_full_solution_after_attempt",
            reason="hint_ladder_exhausted",
        )

    # ------------------------------------------------------------------
    # 5. Fallback branches for non-ladder intents.
    # ------------------------------------------------------------------
    if signals.intent == IntentType.CHECK_WORK:
        return Action(action_type=ActionType.EXPLAIN, move="verify_and_explain", reason="check_work_fallback")

    if signals.intent == IntentType.CONCEPT_EXPLAIN:
        return Action(
            action_type=ActionType.EXPLAIN, move="concept_explanation", reason="concept_explain_fallback"
        )

    if signals.intent == IntentType.EXAM_PREP:
        return Action(action_type=ActionType.QUESTION, move="practice_question", reason="exam_prep_fallback")

    # ------------------------------------------------------------------
    # 6. Default fallback (e.g. off_topic, or any future intent value
    #    that isn't explicitly handled above).
    # ------------------------------------------------------------------
    return Action(action_type=ActionType.EXPLAIN, move="redirect_to_subject", reason="default_fallback")
