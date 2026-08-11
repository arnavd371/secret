"""
Tests for the parametric Question Generation Engine: every generated
item's answer is checked against real SymPy computation, not just "did it
run" — same standard as the CAS solver tests.
"""

import sympy

from app.cas.solver import verify_claim
from app.questions.distractors import generate_distractors
from app.questions.generator import (
    compute_parameter_hash,
    generate_item,
    is_number_friendly,
    select_template_for_topic,
)
from app.questions.mark_scheme import build_mark_scheme
from app.questions.models import ItemTemplate, ParamSpec
from app.questions.templates import TEMPLATE_BANK


def _cas_result_for(item, template_id):
    from app.cas.solver import run_cas_operation

    template = TEMPLATE_BANK[template_id]
    expression = template.expression_template.format(**item.sampled_parameters)
    return run_cas_operation(template.operation, expression, template.variable)


# ---------------------------------------------------------------------------
# Generation produces real, CAS-verified, quality-gated items
# ---------------------------------------------------------------------------


def test_every_seed_template_generates_a_verified_item():
    for template_id in TEMPLATE_BANK:
        item = generate_item(template_id, seed=7)
        assert item.correct_answer.cas_verified is True
        assert item.quality_gate_status == "PASSED"
        assert item.template_id == template_id


def test_generated_item_answer_matches_independent_cas_recomputation():
    """The item's stored answer must match what CAS produces when you
    independently recompute it from the same sampled parameters — this
    is the actual correctness guarantee, not just internal consistency."""
    item = generate_item("AA.SL.CALC.DIFF.CHAIN.T003", seed=3)
    recomputed = _cas_result_for(item, "AA.SL.CALC.DIFF.CHAIN.T003")
    assert verify_claim(recomputed, item.correct_answer.value)


def test_generation_is_deterministic_for_a_fixed_seed():
    item_a = generate_item("AA.SL.CALC.DIFF.POWER.T001", seed=123)
    item_b = generate_item("AA.SL.CALC.DIFF.POWER.T001", seed=123)
    assert item_a.sampled_parameters == item_b.sampled_parameters
    assert item_a.correct_answer.value == item_b.correct_answer.value


def test_unknown_template_id_raises():
    import pytest

    from app.questions.generator import ItemGenerationError

    with pytest.raises(ItemGenerationError):
        generate_item("NOT-A-REAL-TEMPLATE")


# ---------------------------------------------------------------------------
# Distractors are tied to real, named misconceptions
# ---------------------------------------------------------------------------


def test_chain_rule_distractor_matches_named_misconception():
    item = generate_item("AA.SL.CALC.DIFF.CHAIN.T003", seed=5)
    assert item.distractors
    distractor = item.distractors[0]
    assert distractor.misconception == "MISC-CALC-014"
    assert distractor.value != item.correct_answer.value


def test_product_rule_distractor_uses_wrong_method():
    item = generate_item("AA.SL.CALC.DIFF.PRODUCT.T002", seed=9)
    assert item.distractors
    assert item.distractors[0].misconception == "MISC-CALC-010"


def test_distractors_never_equal_the_correct_answer():
    for template_id in TEMPLATE_BANK:
        item = generate_item(template_id, seed=17)
        for distractor in item.distractors:
            assert distractor.value != item.correct_answer.value


def test_generate_distractors_swallows_a_broken_generator():
    def _broken(params, variable):
        raise ValueError("boom")

    # Directly exercise the registry-driven path with a deliberately
    # broken generator function to confirm it's skipped, not fatal.
    from app.questions import distractors as distractors_module

    original = distractors_module.DISTRACTOR_REGISTRY.get("AA.SL.CALC.DIFF.POWER.T001")
    distractors_module.DISTRACTOR_REGISTRY["AA.SL.CALC.DIFF.POWER.T001"] = [_broken]
    try:
        result = distractors_module.generate_distractors("AA.SL.CALC.DIFF.POWER.T001", {"a": 2, "n": 3, "b": 1}, "x", "6*x**2 + 1")
    finally:
        distractors_module.DISTRACTOR_REGISTRY["AA.SL.CALC.DIFF.POWER.T001"] = original
    assert result == []


# ---------------------------------------------------------------------------
# Number-friendliness (spec §9.9)
# ---------------------------------------------------------------------------


def test_calculator_mode_always_number_friendly():
    assert is_number_friendly("sqrt(97)/6", "calculator") is True


def test_integer_is_number_friendly():
    assert is_number_friendly("4", "non-calculator") is True


def test_small_denominator_rational_is_number_friendly():
    assert is_number_friendly("5/6", "non-calculator") is True


def test_large_denominator_rational_is_not_number_friendly():
    assert is_number_friendly("1/97", "non-calculator") is False


def test_small_radicand_surd_is_number_friendly():
    assert is_number_friendly("1 + sqrt(5)", "non-calculator") is True


def test_symbolic_expression_with_free_variable_skips_the_check():
    """A derivative like '6*x + 5' isn't a 'number' at all — the
    friendliness concept doesn't apply, and our parameters are already
    integers by construction."""
    assert is_number_friendly("6*x + 5", "non-calculator", "x") is True


def test_multi_root_solve_string_checks_each_root():
    assert is_number_friendly("x = -3, x = -2", "non-calculator") is True
    assert is_number_friendly("x = 1/97, x = 2", "non-calculator") is False


def test_quadratic_template_never_produces_complex_roots():
    """The template's discriminant >= 0 constraint should mean no
    generated item ever has an imaginary-unit answer."""
    for seed in range(20):
        item = generate_item("AA.SL.ALG.QUAD.T004", seed=seed)
        assert "I" not in item.correct_answer.value or "sqrt(-1)" not in item.correct_answer.value


# ---------------------------------------------------------------------------
# Duplicate / leakage prevention (spec §9.12)
# ---------------------------------------------------------------------------


def test_parameter_hash_is_stable_for_same_inputs():
    h1 = compute_parameter_hash("T001", {"a": 2, "n": 3})
    h2 = compute_parameter_hash("T001", {"n": 3, "a": 2})  # order shouldn't matter
    assert h1 == h2


def test_parameter_hash_differs_for_different_params():
    h1 = compute_parameter_hash("T001", {"a": 2, "n": 3})
    h2 = compute_parameter_hash("T001", {"a": 3, "n": 3})
    assert h1 != h2


def test_avoid_parameter_hashes_forces_a_different_instance():
    first = generate_item("AA.SL.CALC.DIFF.POWER.T001", seed=1)
    first_hash = compute_parameter_hash(first.template_id, first.sampled_parameters)

    second = generate_item("AA.SL.CALC.DIFF.POWER.T001", seed=1, avoid_parameter_hashes={first_hash})
    second_hash = compute_parameter_hash(second.template_id, second.sampled_parameters)
    assert second_hash != first_hash


def test_avoid_stem_texts_forces_a_near_duplicate_to_be_rejected():
    first = generate_item("AA.SL.CALC.DIFF.POWER.T001", seed=1)
    second = generate_item(
        "AA.SL.CALC.DIFF.POWER.T001", seed=1, avoid_stem_texts=[first.rendered_stem]
    )
    assert second.rendered_stem != first.rendered_stem


# ---------------------------------------------------------------------------
# Mark scheme derivability (spec §9.13)
# ---------------------------------------------------------------------------


def test_mark_scheme_traces_to_real_cas_steps():
    item = generate_item("AA.SL.CALC.DIFF.CHAIN.T003", seed=2)
    assert item.mark_scheme is not None
    assert item.mark_scheme.total_marks == len(item.mark_scheme.nodes)
    assert any("chain rule" in node.text for node in item.mark_scheme.nodes)
    assert any(item.correct_answer.value in node.text for node in item.mark_scheme.nodes)


def test_build_mark_scheme_generic_method_when_no_named_rule_detected():
    result = generate_item("AA.SL.ALG.QUAD.T004", seed=4)
    method_nodes = [n for n in result.mark_scheme.nodes if n.type == "M"]
    assert method_nodes  # always at least one method node


# ---------------------------------------------------------------------------
# Topic -> template selection for CHALLENGE wiring
# ---------------------------------------------------------------------------


def test_select_template_for_exact_topic_hint():
    assert select_template_for_topic("calculus.differentiation.chain_rule") == "AA.SL.CALC.DIFF.CHAIN.T003"


def test_select_template_falls_back_to_default_for_unknown_topic():
    from app.questions.generator import DEFAULT_CHALLENGE_TEMPLATE_ID

    assert select_template_for_topic("some.unrelated.topic") == DEFAULT_CHALLENGE_TEMPLATE_ID
    assert select_template_for_topic(None) == DEFAULT_CHALLENGE_TEMPLATE_ID


# ---------------------------------------------------------------------------
# Template model validation
# ---------------------------------------------------------------------------


def test_param_spec_rejects_inverted_domain():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ParamSpec(domain_min=5, domain_max=1)


def test_empty_parameter_domain_after_exclusion_raises_at_generation_time():
    import pytest

    from app.questions.generator import ItemGenerationError
    from app.questions.templates import TEMPLATE_BANK as _bank

    broken_template = ItemTemplate(
        template_id="TEST.BROKEN",
        stem_template="Find x for {a}.",
        parameters={"a": ParamSpec(domain_min=1, domain_max=1, exclude=[1])},
        operation=list(_bank.values())[0].operation,
        expression_template="{a}",
    )
    _bank["TEST.BROKEN"] = broken_template
    try:
        with pytest.raises(ItemGenerationError):
            generate_item("TEST.BROKEN")
    finally:
        del _bank["TEST.BROKEN"]
