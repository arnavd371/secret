from app.examiner.models import StepType
from app.examiner.segmentation import segment_work


def test_splits_on_newlines_and_semicolons():
    steps = segment_work("u = x**2\nv = sin(x); w = cos(x)")
    assert [s.raw_text for s in steps] == ["u = x**2", "v = sin(x)", "w = cos(x)"]


def test_blank_lines_are_dropped():
    steps = segment_work("u = x**2\n\n\nv = sin(x)")
    assert len(steps) == 2


def test_algebraic_step_gets_normalized_expression():
    """A single-line submission is, by definition, also the last line, so
    it's classified FINAL_ANSWER (see test_last_algebraic_step_is_
    classified_final_answer below) — use a non-last step here to isolate
    the normalized-expression behavior specifically."""
    steps = segment_work("u_prime = 2*x\ntherefore that's the derivative")
    assert steps[0].step_type == StepType.ALGEBRAIC_MANIPULATION
    assert steps[0].normalized_expr == "2*x"


def test_free_text_reasoning_step_has_no_normalized_expression():
    steps = segment_work("since the exponent is even, the function is symmetric")
    assert steps[0].step_type == StepType.JUSTIFICATION
    assert steps[0].normalized_expr is None


def test_last_algebraic_step_is_classified_final_answer():
    steps = segment_work("u = x**2\ntherefore dy/dx = 2*x")
    assert steps[-1].step_type == StepType.FINAL_ANSWER


def test_middle_algebraic_step_is_not_final_answer():
    steps = segment_work("u = x**2\nv = sin(x)\ntherefore dy/dx = 2*x*sin(x) + x**2*cos(x)")
    assert steps[0].step_type == StepType.ALGEBRAIC_MANIPULATION
    assert steps[1].step_type == StepType.ALGEBRAIC_MANIPULATION
    assert steps[2].step_type == StepType.FINAL_ANSWER


def test_restatement_of_given_is_detected_when_matching_given_expression():
    steps = segment_work("y = x**2 * sin(x)\ntherefore dy/dx = 2*x*sin(x)", given_expression="x**2*sin(x)")
    assert steps[0].step_type == StepType.RESTATEMENT_OF_GIVEN


def test_unparseable_algebra_falls_back_to_justification():
    steps = segment_work("this = not valid math ###")
    assert steps[0].step_type == StepType.JUSTIFICATION
    assert steps[0].normalized_expr is None
