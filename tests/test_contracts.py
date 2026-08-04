"""
Validation tests for the typed contracts: bad data must fail loudly at the
boundary (a ValidationError), never propagate as e.g. a hint level of 7 or
a confidence of 1.5.
"""

import pytest
from pydantic import ValidationError

from app.models.contracts import (
    Action,
    ActionType,
    AssessmentMode,
    DecisionSignals,
    FrustrationLevel,
    IntegrityRisk,
    IntentResult,
    IntentType,
)


def test_intent_result_confidence_must_be_0_to_1():
    with pytest.raises(ValidationError):
        IntentResult(intent=IntentType.SOLVE_REQUEST, confidence=1.5, subject="math_aa")

    with pytest.raises(ValidationError):
        IntentResult(intent=IntentType.SOLVE_REQUEST, confidence=-0.1, subject="math_aa")

    # boundary values are valid
    IntentResult(intent=IntentType.SOLVE_REQUEST, confidence=0.0, subject="math_aa")
    IntentResult(intent=IntentType.SOLVE_REQUEST, confidence=1.0, subject="math_aa")


def test_decision_signals_hint_ladder_level_bounds():
    base_kwargs = dict(
        intent=IntentType.SOLVE_REQUEST,
        mastery_estimate=0.5,
        assessment_mode=AssessmentMode.PRACTICE,
        integrity_risk=IntegrityRisk.LOW,
        attempt_count=1,
        frustration_signal=FrustrationLevel.NONE,
    )
    with pytest.raises(ValidationError):
        DecisionSignals(**base_kwargs, hint_ladder_level=5)
    with pytest.raises(ValidationError):
        DecisionSignals(**base_kwargs, hint_ladder_level=-1)

    # boundary values are valid
    DecisionSignals(**base_kwargs, hint_ladder_level=0)
    DecisionSignals(**base_kwargs, hint_ladder_level=4)


def test_decision_signals_attempt_count_cannot_be_negative():
    with pytest.raises(ValidationError):
        DecisionSignals(
            intent=IntentType.SOLVE_REQUEST,
            mastery_estimate=0.5,
            assessment_mode=AssessmentMode.PRACTICE,
            integrity_risk=IntegrityRisk.LOW,
            attempt_count=-1,
            frustration_signal=FrustrationLevel.NONE,
            hint_ladder_level=0,
        )


def test_hint_action_requires_level():
    with pytest.raises(ValueError):
        Action(action_type=ActionType.HINT, reason="missing level")

    # a valid HINT action must specify a level in [1, 4]
    Action(action_type=ActionType.HINT, level=2, reason="ok")


def test_hint_action_level_must_be_1_to_4():
    with pytest.raises(ValidationError):
        Action(action_type=ActionType.HINT, level=5, reason="too high")
    with pytest.raises(ValidationError):
        Action(action_type=ActionType.HINT, level=0, reason="zero not allowed on HINT")


def test_non_hint_action_cannot_carry_a_level():
    with pytest.raises(ValueError):
        Action(action_type=ActionType.EXPLAIN, level=2, reason="explain shouldn't have a level")


def test_action_requires_nonempty_reason():
    with pytest.raises(ValidationError):
        Action(action_type=ActionType.EXPLAIN, reason="")


def test_action_offer_accepts_string_or_list():
    single = Action(action_type=ActionType.REFUSE, offer="concept_explanation", reason="ok")
    assert single.offer == "concept_explanation"

    multi = Action(
        action_type=ActionType.REFUSE, offer=["concept_explanation", "analog_practice_problem"], reason="ok"
    )
    assert multi.offer == ["concept_explanation", "analog_practice_problem"]
