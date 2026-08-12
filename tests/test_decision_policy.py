"""
Full branch coverage of decide_pedagogical_action, including branch-
ordering edge cases (a hard gate must win even when a later branch's
condition is also satisfied).

This is the cheapest, highest-value test suite in the system: the policy
is pure, so every test is a plain call + assert, no mocking required.
Cases are traced directly to the spec §1.5 pseudocode and decision table.
"""

from app.ia_supervisor.models import IAStage
from app.models.contracts import (
    ActionType,
    AssessmentMode,
    DecisionSignals,
    FrustrationLevel,
    IntegrityRisk,
    IntentType,
)
from app.policy.decision import (
    HIGH_MASTERY_THRESHOLD,
    MID_ATTEMPT_HINT_CAP,
    decide_pedagogical_action,
)
from app.models.contracts import MAX_HINT_LADDER_LEVEL


def _signals(**overrides) -> DecisionSignals:
    defaults = dict(
        intent=IntentType.SOLVE_REQUEST,
        mastery_estimate=0.5,
        assessment_mode=AssessmentMode.PRACTICE,
        integrity_risk=IntegrityRisk.LOW,
        attempt_count=1,
        frustration_signal=FrustrationLevel.NONE,
        hint_ladder_level=0,
    )
    defaults.update(overrides)
    return DecisionSignals(**defaults)


# ---------------------------------------------------------------------------
# 1. Hard gates
# ---------------------------------------------------------------------------


def test_integrity_risk_high_always_refuses():
    action = decide_pedagogical_action(_signals(integrity_risk=IntegrityRisk.HIGH))
    assert action.action_type == ActionType.REFUSE
    assert action.offer == "concept_explanation"


def test_integrity_risk_high_wins_over_mastery_shortcut():
    """The critical ordering guarantee: even a signal set that would
    otherwise produce CHALLENGE (high mastery, first attempt) must still
    be refused when integrity_risk is high."""
    action = decide_pedagogical_action(
        _signals(integrity_risk=IntegrityRisk.HIGH, mastery_estimate=0.99, attempt_count=1)
    )
    assert action.action_type == ActionType.REFUSE
    assert action.reason == "integrity_risk_high"


def test_integrity_risk_high_wins_over_frustration_override():
    action = decide_pedagogical_action(
        _signals(integrity_risk=IntegrityRisk.HIGH, frustration_signal=FrustrationLevel.HIGH)
    )
    assert action.action_type == ActionType.REFUSE


def test_live_exam_simulation_always_refuses_even_with_low_integrity_risk():
    action = decide_pedagogical_action(
        _signals(assessment_mode=AssessmentMode.LIVE_EXAM_SIMULATION, integrity_risk=IntegrityRisk.LOW)
    )
    assert action.action_type == ActionType.REFUSE
    assert action.offer == "strategy_coaching_only"
    assert action.reason == "live_exam_simulation"


def test_graded_take_home_with_medium_risk_refuses():
    action = decide_pedagogical_action(
        _signals(assessment_mode=AssessmentMode.GRADED_TAKE_HOME, integrity_risk=IntegrityRisk.MEDIUM)
    )
    assert action.action_type == ActionType.REFUSE
    assert action.offer == ["concept_explanation", "analog_practice_problem"]
    assert action.reason == "graded_take_home_elevated_risk"


def test_graded_take_home_with_high_risk_refuses():
    action = decide_pedagogical_action(
        _signals(assessment_mode=AssessmentMode.GRADED_TAKE_HOME, integrity_risk=IntegrityRisk.HIGH)
    )
    assert action.action_type == ActionType.REFUSE


def test_graded_take_home_with_low_risk_does_not_hard_gate():
    action = decide_pedagogical_action(
        _signals(assessment_mode=AssessmentMode.GRADED_TAKE_HOME, integrity_risk=IntegrityRisk.LOW, attempt_count=0)
    )
    assert action.action_type != ActionType.REFUSE


# ---------------------------------------------------------------------------
# 2. IA/EE routing
# ---------------------------------------------------------------------------


def test_ia_ee_help_with_detected_ghostwriting_request_is_refused():
    action = decide_pedagogical_action(
        _signals(intent=IntentType.IA_EE_HELP, ia_ghostwriting_request_detected=True)
    )
    assert action.action_type == ActionType.REFUSE
    assert action.offer == "ia_methodology_coaching"
    assert action.reason == "ia_ghostwriting_guard_tripped"


def test_ia_ee_help_on_a_completed_project_is_refused():
    action = decide_pedagogical_action(
        _signals(intent=IntentType.IA_EE_HELP, ia_project_complete=True)
    )
    assert action.action_type == ActionType.REFUSE
    assert action.reason == "ia_project_already_complete"


def test_ia_ee_help_with_no_guard_tripped_gets_real_bounded_coaching():
    action = decide_pedagogical_action(
        _signals(intent=IntentType.IA_EE_HELP, ia_stage=IAStage.METHODOLOGY)
    )
    assert action.action_type == ActionType.EXPLAIN
    assert action.move == "ia_methodology_coaching"
    assert action.reason == "ia_supervisor_coaching_allowed"


def test_ia_ee_help_with_no_stage_resolved_defaults_to_topic_coaching():
    action = decide_pedagogical_action(_signals(intent=IntentType.IA_EE_HELP))
    assert action.move == "ia_topic_coaching"


def test_ia_ee_help_routes_before_mastery_shortcut():
    """A student with high mastery asking for IA help must not be
    CHALLENGE'd — IA routing takes priority over the mastery shortcut,
    regardless of which IA branch (refuse or coach) it resolves to."""
    action = decide_pedagogical_action(
        _signals(
            intent=IntentType.IA_EE_HELP,
            mastery_estimate=0.99,
            attempt_count=1,
            ia_ghostwriting_request_detected=True,
        )
    )
    assert action.action_type == ActionType.REFUSE
    assert action.action_type != ActionType.CHALLENGE


# ---------------------------------------------------------------------------
# 3. Frustration override
# ---------------------------------------------------------------------------


def test_high_frustration_overrides_normal_ladder():
    action = decide_pedagogical_action(
        _signals(frustration_signal=FrustrationLevel.HIGH, attempt_count=3, hint_ladder_level=2)
    )
    assert action.action_type == ActionType.SUPPORTIVE_SCAFFOLD
    assert action.move == "worked_example_with_fade"
    assert action.tone == "reassuring"
    assert action.reduce_difficulty is True


def test_high_frustration_wins_over_mastery_shortcut():
    action = decide_pedagogical_action(
        _signals(frustration_signal=FrustrationLevel.HIGH, mastery_estimate=0.95, attempt_count=1)
    )
    assert action.action_type == ActionType.SUPPORTIVE_SCAFFOLD


def test_mild_frustration_does_not_override_the_ladder():
    """Only 'high' frustration triggers the override per spec — 'mild'
    should fall through to the normal Socratic ladder."""
    action = decide_pedagogical_action(_signals(frustration_signal=FrustrationLevel.MILD, attempt_count=1))
    assert action.action_type == ActionType.QUESTION


# ---------------------------------------------------------------------------
# 4. Mastery-based shortcut
# ---------------------------------------------------------------------------


def test_high_mastery_first_attempt_triggers_challenge():
    action = decide_pedagogical_action(
        _signals(mastery_estimate=HIGH_MASTERY_THRESHOLD, attempt_count=1)
    )
    assert action.action_type == ActionType.CHALLENGE
    assert action.move == "extension_question"


def test_high_mastery_below_threshold_does_not_trigger_challenge():
    action = decide_pedagogical_action(
        _signals(mastery_estimate=HIGH_MASTERY_THRESHOLD - 0.01, attempt_count=1)
    )
    assert action.action_type != ActionType.CHALLENGE


def test_high_mastery_on_a_later_attempt_does_not_trigger_challenge():
    """Per the pseudocode, the mastery shortcut requires attempt_count == 1
    exactly — a student re-attempting the same problem a third time is not
    treated as a fresh high-mastery encounter."""
    action = decide_pedagogical_action(_signals(mastery_estimate=0.99, attempt_count=3))
    assert action.action_type != ActionType.CHALLENGE


def test_high_mastery_non_solve_request_does_not_trigger_challenge():
    action = decide_pedagogical_action(
        _signals(intent=IntentType.CHECK_WORK, mastery_estimate=0.99, attempt_count=1)
    )
    assert action.action_type == ActionType.EXPLAIN
    assert action.move == "error_localization_explanation"


# ---------------------------------------------------------------------------
# 5. Core Socratic ladder (solve_request)
# ---------------------------------------------------------------------------


def test_first_attempt_is_socratic_question():
    action = decide_pedagogical_action(_signals(attempt_count=0))
    assert action.action_type == ActionType.QUESTION
    assert action.move == "diagnostic_probe"


def test_second_attempt_is_still_socratic_question():
    action = decide_pedagogical_action(_signals(attempt_count=1))
    assert action.action_type == ActionType.QUESTION


def test_attempt_two_gives_hint_level_one_from_fresh_ladder():
    action = decide_pedagogical_action(_signals(attempt_count=2, hint_ladder_level=0))
    assert action.action_type == ActionType.HINT
    assert action.level == 1


def test_attempt_three_escalates_hint_level_capped_at_two():
    """attempt_count in (2,3) is capped at level 2 even if hint_ladder_level
    would otherwise push it higher."""
    action = decide_pedagogical_action(_signals(attempt_count=3, hint_ladder_level=3))
    assert action.action_type == ActionType.HINT
    assert action.level == MID_ATTEMPT_HINT_CAP


def test_attempt_four_plus_escalates_up_to_ladder_max():
    action = decide_pedagogical_action(_signals(attempt_count=4, hint_ladder_level=2))
    assert action.action_type == ActionType.HINT
    assert action.level == 3

    action = decide_pedagogical_action(_signals(attempt_count=5, hint_ladder_level=3))
    assert action.level == MAX_HINT_LADDER_LEVEL


def test_attempt_four_plus_never_exceeds_ladder_max():
    action = decide_pedagogical_action(_signals(attempt_count=9, hint_ladder_level=MAX_HINT_LADDER_LEVEL))
    assert action.level == MAX_HINT_LADDER_LEVEL


# ---------------------------------------------------------------------------
# 6. Fallback branches
# ---------------------------------------------------------------------------


def test_check_work_falls_back_to_error_localization_explanation():
    action = decide_pedagogical_action(_signals(intent=IntentType.CHECK_WORK))
    assert action.action_type == ActionType.EXPLAIN
    assert action.move == "error_localization_explanation"


def test_concept_explain_falls_back_to_direct_explanation():
    action = decide_pedagogical_action(_signals(intent=IntentType.CONCEPT_EXPLAIN))
    assert action.action_type == ActionType.EXPLAIN
    assert action.move == "direct_explanation"


def test_exam_prep_with_a_due_review_asks_a_retrieval_practice_question():
    action = decide_pedagogical_action(_signals(intent=IntentType.EXAM_PREP, has_due_review=True))
    assert action.action_type == ActionType.QUESTION
    assert action.move == "retrieval_practice"
    assert action.reason == "exam_prep_review_due"


def test_exam_prep_with_nothing_due_falls_back_to_a_general_explanation():
    action = decide_pedagogical_action(_signals(intent=IntentType.EXAM_PREP, has_due_review=False))
    assert action.action_type == ActionType.EXPLAIN
    assert action.move == "general_response"
    assert action.reason == "exam_prep_no_review_due"


# ---------------------------------------------------------------------------
# 7. Default fallback
# ---------------------------------------------------------------------------


def test_general_chat_uses_default_fallback():
    action = decide_pedagogical_action(_signals(intent=IntentType.GENERAL_CHAT))
    assert action.action_type == ActionType.EXPLAIN
    assert action.move == "general_response"


# ---------------------------------------------------------------------------
# Purity: calling twice with equal input gives an equal (though not
# necessarily identical-object) result, and the function does not mutate
# its input.
# ---------------------------------------------------------------------------


def test_policy_is_pure():
    signals = _signals(attempt_count=2, hint_ladder_level=1)
    snapshot = signals.model_copy()
    result1 = decide_pedagogical_action(signals)
    result2 = decide_pedagogical_action(signals)
    assert signals == snapshot  # unmutated
    assert result1 == result2
