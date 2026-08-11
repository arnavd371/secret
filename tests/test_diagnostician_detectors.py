"""
Real-math assertions against the pattern detectors — every hypothesis
asserted here is independently checkable by hand (or against Phase 3's
own distractor generators, which these intentionally generalize).
"""

from app.cas.models import CASOperation
from app.diagnostician.detectors import detect_pattern_misconception, generate_hypotheses


def test_dropped_term_hypothesis_for_a_sum_of_two_terms():
    # d/dx[3x^2 + 5x] = 6x + 5; dropping the linear term entirely gives 6x.
    hypotheses = generate_hypotheses(CASOperation.DIFFERENTIATE, "3*x**2 + 5*x", "x")
    ids = {h[0] for h in hypotheses}
    assert "MISC-CALC-GEN-DROPPED-TERM" in ids


def test_dropped_term_match_detects_the_misconception():
    match = detect_pattern_misconception(CASOperation.DIFFERENTIATE, "3*x**2 + 5*x", "x", "6*x")
    assert match is not None
    misconception_id, evidence = match
    assert misconception_id == "MISC-CALC-GEN-DROPPED-TERM"
    assert "6*x" in evidence


def test_product_rule_wrong_method_detected():
    # d/dx[x^2 sin(x)] correct = x^2 cos(x) + 2x sin(x); f'g' = 2x*cos(x).
    match = detect_pattern_misconception(CASOperation.DIFFERENTIATE, "x**2 * sin(x)", "x", "2*x*cos(x)")
    assert match is not None
    assert match[0] == "MISC-CALC-010"


def test_product_rule_wrong_method_not_detected_for_the_correct_answer():
    match = detect_pattern_misconception(
        CASOperation.DIFFERENTIATE, "x**2 * sin(x)", "x", "2*x*sin(x) + x**2*cos(x)"
    )
    assert match is None


def test_chain_rule_outer_only_detected():
    # d/dx[(2x+1)^5] correct = 10(2x+1)^4; outer-only wrong = 5(2x+1)^4.
    match = detect_pattern_misconception(CASOperation.DIFFERENTIATE, "(2*x + 1)**5", "x", "5*(2*x+1)**4")
    assert match is not None
    assert match[0] == "MISC-CALC-014"


def test_chain_rule_hypothesis_not_generated_for_plain_power_rule():
    # x^5 has no real "inner function" (derivative of the base is 1), so
    # the outer-only error is indistinguishable from a correct answer —
    # no hypothesis should be generated at all.
    hypotheses = generate_hypotheses(CASOperation.DIFFERENTIATE, "x**5", "x")
    ids = {h[0] for h in hypotheses}
    assert "MISC-CALC-014" not in ids


def test_quadratic_sign_error_detected():
    # x^2 - 5x + 6 = 0 -> correct roots 2, 3. Sign error (+b) gives -2, -3.
    match = detect_pattern_misconception(CASOperation.SOLVE, "x**2 - 5*x + 6 = 0", "x", "x = -2, x = -3")
    assert match is not None
    assert match[0] == "MISC-ALG-003"


def test_quadratic_sign_error_root_order_does_not_matter():
    match = detect_pattern_misconception(CASOperation.SOLVE, "x**2 - 5*x + 6 = 0", "x", "x = -3, x = -2")
    assert match is not None
    assert match[0] == "MISC-ALG-003"


def test_quadratic_sign_error_not_detected_for_correct_roots():
    match = detect_pattern_misconception(CASOperation.SOLVE, "x**2 - 5*x + 6 = 0", "x", "x = 2, x = 3")
    assert match is None


def test_no_hypothesis_generated_for_a_linear_equation():
    # Degree 1, not 2 — the quadratic-formula sign-error pattern doesn't apply.
    hypotheses = generate_hypotheses(CASOperation.SOLVE, "2*x + 4 = 0", "x")
    assert hypotheses == []


def test_unrelated_wrong_answer_matches_nothing():
    match = detect_pattern_misconception(CASOperation.DIFFERENTIATE, "x**2 * sin(x)", "x", "999")
    assert match is None


def test_no_detectors_for_integrate_operation():
    hypotheses = generate_hypotheses(CASOperation.INTEGRATE, "x**2", "x")
    assert hypotheses == []
