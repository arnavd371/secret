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
    IntegrityRisk,
    IntentResult,
    IntentType,
)


def test_intent_result_confidence_must_be_0_to_1():
    with pytest.raises(ValidationError):
        IntentResult(intent=IntentType.PRACTICE, confidence=1.5, subject="Math AA")

    with pytest.raises(ValidationError):
        IntentResult(intent=IntentType.PRACTICE, confidence=-0.1, subject="Math AA")

    # boundary values are valid
    IntentResult(intent=IntentType.PRACTICE, confidence=0.0, subject="Math AA")
    IntentResult(intent=IntentType.PRACTICE, confidence=1.0, subject="Math AA")


def test_decision_signals_hint_ladder_level_bounds():
    base_kwargs = dict(
        intent=IntentType.PRACTICE,
        mastery_estimate=0.5,
        assessment_mode=AssessmentMode.PRACTICE,
        integrity_risk=IntegrityRisk.NONE,
        attempt_count=1,
        frustration_signal=False,
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
            intent=IntentType.PRACTICE,
            mastery_estimate=0.5,
            assessment_mode=AssessmentMode.PRACTICE,
            integrity_risk=IntegrityRisk.NONE,
            attempt_count=-1,
            frustration_signal=False,
            hint_ladder_level=0,
        )


def test_hint_action_requires_level():
    with pytest.raises(ValueError):
        Action(action_type=ActionType.HINT, reason="missing level")

    # a valid HINT action must specify a level
    Action(action_type=ActionType.HINT, level=2, reason="ok")


def test_action_level_must_be_0_to_4():
    with pytest.raises(ValidationError):
        Action(action_type=ActionType.HINT, level=5, reason="too high")
    with pytest.raises(ValidationError):
        Action(action_type=ActionType.HINT, level=-1, reason="negative")


def test_action_requires_nonempty_reason():
    with pytest.raises(ValidationError):
        Action(action_type=ActionType.EXPLAIN, reason="")
