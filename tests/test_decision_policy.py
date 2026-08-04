"""
Full branch coverage of decide_pedagogical_action, including branch-
ordering edge cases (a hard gate must win even when a later branch's
condition is also satisfied).

This is the cheapest, highest-value test suite in the system: the policy
is pure, so every test is a plain call + assert, no mocking required.
"""

from app.models.contracts import (
    Action,
    ActionType,
    AssessmentMode,
    DecisionSignals,
    IntegrityRisk,
    IntentType,
)
from app.policy.decision import (
    HIGH_MASTERY_THRESHOLD,
    MAX_HINT_LEVEL,
    decide_pedagogical_action,
)


def _signals(**overrides) -> DecisionSignals:
    defaults = dict(
        intent=IntentType.PRACTICE,
        mastery_estimate=0.5,
        assessment_mode=AssessmentMode.PRACTICE,
        integrity_risk=IntegrityRisk.NONE,
        attempt_count=1,
        frustration_signal=False,
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


def test_integrity_risk_high_wins_over_mastery_shortcut():
    """The critical ordering guarantee: even a signal set that would
    otherwise produce CHALLENGE (high mastery, practice intent) must
    still be refused when integrity_risk is high."""
    action = decide_pedagogical_action(
        _signals(
            integrity_risk=IntegrityRisk.HIGH,
            mastery_estimate=0.99,
            intent=IntentType.PRACTICE,
        )
    )
    assert action.action_type == ActionType.REFUSE
    assert action.reason == "integrity_risk_high"


def test_integrity_risk_high_wins_over_frustration_override():
    action = decide_pedagogical_action(
        _signals(integrity_risk=IntegrityRisk.HIGH, frustration_signal=True)
    )
    assert action.action_type == ActionType.REFUSE


def test_live_exam_simulation_always_refuses_even_with_no_integrity_risk():
    action = decide_pedagogical_action(
        _signals(assessment_mode=AssessmentMode.LIVE_EXAM_SIMULATION, integrity_risk=IntegrityRisk.NONE)
    )
    assert action.action_type == ActionType.REFUSE
    assert action.reason == "live_exam_simulation"


def test_graded_take_home_with_medium_risk_refuses():
    action = decide_pedagogical_action(
        _signals(assessment_mode=AssessmentMode.GRADED_TAKE_HOME, integrity_risk=IntegrityRisk.MEDIUM)
    )
    assert action.action_type == ActionType.REFUSE
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


def test_graded_take_home_with_no_risk_does_not_hard_gate():
    action = decide_pedagogical_action(
        _signals(assessment_mode=AssessmentMode.GRADED_TAKE_HOME, integrity_risk=IntegrityRisk.NONE, attempt_count=0)
    )
    assert action.action_type != ActionType.REFUSE


# ---------------------------------------------------------------------------
# 2. Frustration override
# ---------------------------------------------------------------------------


def test_frustration_overrides_normal_ladder():
    action = decide_pedagogical_action(_signals(frustration_signal=True, attempt_count=3, hint_ladder_level=2))
    assert action.action_type == ActionType.SUPPORTIVE_SCAFFOLD
    assert action.reason == "frustration_override"


def test_frustration_wins_over_mastery_shortcut():
    action = decide_pedagogical_action(
        _signals(frustration_signal=True, mastery_estimate=0.95, intent=IntentType.PRACTICE)
    )
    assert action.action_type == ActionType.SUPPORTIVE_SCAFFOLD


# ---------------------------------------------------------------------------
# 3. Mastery-based shortcut
# ---------------------------------------------------------------------------


def test_high_mastery_practice_triggers_challenge():
    action = decide_pedagogical_action(
        _signals(mastery_estimate=HIGH_MASTERY_THRESHOLD, intent=IntentType.PRACTICE)
    )
    assert action.action_type == ActionType.CHALLENGE


def test_high_mastery_exam_prep_triggers_challenge():
    action = decide_pedagogical_action(
        _signals(mastery_estimate=0.9, intent=IntentType.EXAM_PREP)
    )
    assert action.action_type == ActionType.CHALLENGE


def test_high_mastery_below_threshold_does_not_trigger_challenge():
    action = decide_pedagogical_action(
        _signals(mastery_estimate=HIGH_MASTERY_THRESHOLD - 0.01, intent=IntentType.PRACTICE, attempt_count=0)
    )
    assert action.action_type != ActionType.CHALLENGE


def test_high_mastery_check_work_does_not_trigger_challenge():
    """CHALLENGE only applies to practice/exam_prep intents; a high-mastery
    student asking to check their work still gets the check_work fallback."""
    action = decide_pedagogical_action(
        _signals(mastery_estimate=0.99, intent=IntentType.CHECK_WORK)
    )
    assert action.action_type == ActionType.EXPLAIN
    assert action.move == "verify_and_explain"


# ---------------------------------------------------------------------------
# 4. Core Socratic ladder (practice / hint_request)
# ---------------------------------------------------------------------------


def test_first_attempt_is_always_socratic_question_regardless_of_stale_ladder():
    action = decide_pedagogical_action(
        _signals(intent=IntentType.PRACTICE, attempt_count=0, hint_ladder_level=3)
    )
    assert action.action_type == ActionType.QUESTION
    assert action.move == "socratic_prompt"


def test_attempt_with_ladder_still_zero_stays_socratic():
    action = decide_pedagogical_action(
        _signals(intent=IntentType.PRACTICE, attempt_count=1, hint_ladder_level=0)
    )
    assert action.action_type == ActionType.QUESTION


def test_mid_ladder_gives_hint_at_current_level():
    for level in range(1, MAX_HINT_LEVEL):
        action = decide_pedagogical_action(
            _signals(intent=IntentType.PRACTICE, attempt_count=2, hint_ladder_level=level)
        )
        assert action.action_type == ActionType.HINT
        assert action.level == level
        assert action.offer is None


def test_max_ladder_level_offers_full_solution():
    action = decide_pedagogical_action(
        _signals(intent=IntentType.PRACTICE, attempt_count=4, hint_ladder_level=MAX_HINT_LEVEL)
    )
    assert action.action_type == ActionType.HINT
    assert action.level == MAX_HINT_LEVEL
    assert action.offer == "offer_full_solution_after_attempt"


def test_hint_request_intent_follows_same_ladder_as_practice():
    action = decide_pedagogical_action(
        _signals(intent=IntentType.HINT_REQUEST, attempt_count=2, hint_ladder_level=2)
    )
    assert action.action_type == ActionType.HINT
    assert action.level == 2


def test_hint_request_first_turn_is_still_socratic():
    action = decide_pedagogical_action(
        _signals(intent=IntentType.HINT_REQUEST, attempt_count=0, hint_ladder_level=0)
    )
    assert action.action_type == ActionType.QUESTION


# ---------------------------------------------------------------------------
# 5. Fallback branches
# ---------------------------------------------------------------------------


def test_check_work_intent_falls_back_to_explain():
    action = decide_pedagogical_action(_signals(intent=IntentType.CHECK_WORK, mastery_estimate=0.5))
    assert action.action_type == ActionType.EXPLAIN
    assert action.move == "verify_and_explain"


def test_concept_explain_intent_falls_back_to_explain():
    action = decide_pedagogical_action(_signals(intent=IntentType.CONCEPT_EXPLAIN))
    assert action.action_type == ActionType.EXPLAIN
    assert action.move == "concept_explanation"


def test_exam_prep_low_mastery_falls_back_to_question():
    action = decide_pedagogical_action(_signals(intent=IntentType.EXAM_PREP, mastery_estimate=0.4))
    assert action.action_type == ActionType.QUESTION
    assert action.move == "practice_question"


# ---------------------------------------------------------------------------
# 6. Default fallback
# ---------------------------------------------------------------------------


def test_off_topic_intent_uses_default_fallback():
    action = decide_pedagogical_action(_signals(intent=IntentType.OFF_TOPIC))
    assert action.action_type == ActionType.EXPLAIN
    assert action.move == "redirect_to_subject"


# ---------------------------------------------------------------------------
# Purity: calling twice with equal input gives an equal (though not
# necessarily identical-object) result, and the function does not mutate
# its input.
# ---------------------------------------------------------------------------


def test_policy_is_pure():
    signals = _signals(intent=IntentType.PRACTICE, attempt_count=2, hint_ladder_level=2)
    snapshot = signals.model_copy()
    result1 = decide_pedagogical_action(signals)
    result2 = decide_pedagogical_action(signals)
    assert signals == snapshot  # unmutated
    assert result1 == result2
