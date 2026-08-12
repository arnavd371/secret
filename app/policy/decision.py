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

from app.ia_supervisor.models import IAStage
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


_IA_COACHING_MOVE_BY_STAGE = {
    IAStage.TOPIC_SELECTION: "ia_topic_coaching",
    IAStage.RESEARCH_QUESTION: "ia_research_question_coaching",
    IAStage.METHODOLOGY: "ia_methodology_coaching",
    IAStage.ANALYSIS: "ia_analysis_coaching",
    IAStage.DRAFTING: "ia_drafting_coaching",
    IAStage.REVISION: "ia_revision_coaching",
}


def _route_to_ia_supervisor(signals: DecisionSignals) -> Action:
    """
    Spec §1.5: `if signals.intent == "ia_ee_help": return
    route_to_ia_supervisor(signals) # see 2.10 — never full ghostwriting`.

    All three real signals this branches on — ia_ghostwriting_request_detected,
    ia_project_complete, ia_stage — are resolved by the orchestrator from
    the real IA Supervisor Agent (app/ia_supervisor/, spec §11: guard,
    state machine, disclosure log) before this pure function runs, same
    convention as has_due_review/mastery_estimate above.

    Two hard-refusal cases, checked first because they're non-negotiable
    regardless of stage: a detected ghostwriting request, or a project
    already marked COMPLETE (a genuine terminal state — coaching closes
    once a project is done, it doesn't linger indefinitely). Otherwise,
    real bounded coaching is allowed, routed to the move matching the
    project's actual current stage so the Tutor prompt's constraints
    (app/agents/templates.py's IA coaching block) and the disclosure log
    both reflect what stage of work this was.
    """
    if signals.ia_ghostwriting_request_detected:
        return Action(
            action_type=ActionType.REFUSE,
            offer="ia_methodology_coaching",
            reason="ia_ghostwriting_guard_tripped",
        )
    if signals.ia_project_complete:
        return Action(
            action_type=ActionType.REFUSE,
            offer="ia_methodology_coaching",
            reason="ia_project_already_complete",
        )
    stage = signals.ia_stage or IAStage.TOPIC_SELECTION
    move = _IA_COACHING_MOVE_BY_STAGE.get(stage, "ia_topic_coaching")
    return Action(action_type=ActionType.EXPLAIN, move=move, reason="ia_supervisor_coaching_allowed")


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
