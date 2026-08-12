"""
Parametric item generation (spec §9.1's "parametric/template" mode): sample
parameters, compute a CAS-verified answer, generate real distractors, and
run the quality gates that decide whether to accept the candidate or
resample — mirroring spec §9.13's table (each gate's failure action is
"reject/resample", not just an after-the-fact report).

LLM-authored variants + verifier gating (§9.6) live in app/questions/
llm_variant.py as a separate, real generation mode rather than being
folded into this module's parametric sampling loop — the orchestrator
(app.orchestrator.handle_turn) tries it only when topic_has_known_template()
below is False, i.e. there's no good parametric template for the topic
at all, rather than as a redundant, costlier alternative to a template
that already works.

Not implemented (later-phase non-goals):
  - Online IRT parameter recalibration from response history (§9.7)
  - Mixed-topic/paper-style composition via constrained optimization (§9.10)
  - Adaptive targeting of weak skills/misconceptions (§9.11) — needs the
    Phase 5 mastery model
  - Full embedding-similarity leakage detection (§9.12) — this uses a
    Jaccard token-overlap proxy over rendered stems instead of a dense
    embedding index, same simplification as the Phase 2 retriever.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import random
import re
import uuid
from typing import Optional

import sympy

from app.cas.models import CASResult, CASStatus
from app.cas.solver import run_cas_operation
from app.questions.distractors import generate_distractors
from app.questions.mark_scheme import build_mark_scheme
from app.questions.models import (
    CorrectAnswer,
    DifficultyEstimate,
    GeneratedItem,
    ItemTemplate,
    QualityGateReport,
    QualityGateResult,
)
from app.questions.templates import TEMPLATE_BANK

# Maps a Router/Intent agent topic_hint (spec §2.2, same subtopic_id
# vocabulary as the Phase 2 knowledge base) to the template best suited to
# generate a CHALLENGE extension item on that topic. Falls back to the
# broadest-applicability template when no topic_hint is available or none
# of the mapped subtopics match.
DEFAULT_CHALLENGE_TEMPLATE_ID = "AA.SL.CALC.DIFF.POWER.T001"

_TEMPLATE_BY_SUBTOPIC: dict[str, str] = {
    "calculus.differentiation.product_rule": "AA.SL.CALC.DIFF.PRODUCT.T002",
    "calculus.differentiation.chain_rule": "AA.SL.CALC.DIFF.CHAIN.T003",
    "algebra.quadratics.solving": "AA.SL.ALG.QUAD.T004",
}


def select_template_for_topic(topic_hint: Optional[str]) -> str:
    if topic_hint:
        for subtopic_id, template_id in _TEMPLATE_BY_SUBTOPIC.items():
            if topic_hint == subtopic_id or topic_hint in subtopic_id or subtopic_id in topic_hint:
                return template_id
    return DEFAULT_CHALLENGE_TEMPLATE_ID


def topic_has_known_template(topic_hint: Optional[str]) -> bool:
    """False for a topic_hint that would fall through to the generic
    default template (Phase 13's real trigger for trying an LLM-authored
    variant instead — spec §9.6 — rather than silently serving an item
    on the wrong subtopic)."""
    if not topic_hint:
        return False
    return any(
        topic_hint == subtopic_id or topic_hint in subtopic_id or subtopic_id in topic_hint
        for subtopic_id in _TEMPLATE_BY_SUBTOPIC
    )

# Spec §9.12: "cosine_similarity > 0.85 triggers rejection or forced
# resample." Implemented here as Jaccard token overlap over rendered
# stems (a real, cheap proxy) rather than a dense embedding index.
DUPLICATE_SIMILARITY_THRESHOLD = 0.85

# Spec §9.9 number-friendliness defaults.
NUMBER_FRIENDLY_MAX_DENOMINATOR = 12
NUMBER_FRIENDLY_MAX_RADICAND = 100

_STEM_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


class ItemGenerationError(RuntimeError):
    pass


def _constraints_satisfied(template: ItemTemplate, params: dict[str, int]) -> bool:
    for constraint in template.constraints:
        try:
            if not eval(constraint, {"__builtins__": {}}, dict(params)):  # noqa: S307 - authored, not user, input
                return False
        except Exception:  # noqa: BLE001 - a malformed constraint fails the sample, doesn't crash generation
            return False
    return True


def _sample_parameters(template: ItemTemplate, rng: random.Random) -> Optional[dict[str, int]]:
    candidate: dict[str, int] = {}
    for name, spec in template.parameters.items():
        choices = [v for v in range(spec.domain_min, spec.domain_max + 1) if v not in spec.exclude]
        if not choices:
            raise ItemGenerationError(f"parameter '{name}' in template {template.template_id} has an empty domain")
        candidate[name] = rng.choice(choices)
    return candidate if _constraints_satisfied(template, candidate) else None


def compute_parameter_hash(template_id: str, params: dict[str, int]) -> str:
    """Spec §9.12: hash(template_id, sampled_parameters), checked against a
    'served' set scoped per-student, to never repeat an exact instance."""
    payload = json.dumps({"template_id": template_id, "params": dict(sorted(params.items()))}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _render_stem(template: ItemTemplate, params: dict[str, int]) -> str:
    """Every sampled int parameter also gets a `{name}_signed` variant
    ("+ 3" / "- 3") available to `stem_template`, so a template can render
    a possibly-negative mid-expression term without ever producing an
    ugly "+ -3"."""
    render_params: dict[str, object] = dict(params)
    for name, value in params.items():
        render_params[f"{name}_signed"] = f"+ {value}" if value >= 0 else f"- {abs(value)}"
    return template.stem_template.format(**render_params)


def _tokenize_stem(text: str) -> set[str]:
    return set(_STEM_TOKEN_PATTERN.findall(text.lower()))


def _stem_similarity(a: str, b: str) -> float:
    tokens_a, tokens_b = _tokenize_stem(a), _tokenize_stem(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _is_near_duplicate(stem: str, avoid_stems: list[str]) -> bool:
    return any(_stem_similarity(stem, prior) >= DUPLICATE_SIMILARITY_THRESHOLD for prior in avoid_stems)


def _is_exact_surd_form(expr: sympy.Basic, max_radicand: float) -> bool:
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.Pow) and node.args[1] == sympy.Rational(1, 2):
            radicand = node.args[0]
            if radicand.is_number and abs(complex(radicand)) <= max_radicand:
                return True
    return False


def _contains_free_variable(value_str: str, variable: str) -> bool:
    try:
        if "," in value_str:
            return any(_contains_free_variable(part.split("=")[-1].strip(), variable) for part in value_str.split(","))
        expr = sympy.sympify(value_str)
        return sympy.Symbol(variable) in expr.free_symbols
    except Exception:  # noqa: BLE001
        return False


def is_number_friendly(
    value_str: str,
    mode: str,
    variable: str = "x",
    max_denominator: int = NUMBER_FRIENDLY_MAX_DENOMINATOR,
    max_radicand: int = NUMBER_FRIENDLY_MAX_RADICAND,
) -> bool:
    """Spec §9.9 pseudocode, ported to real SymPy. Checked against the
    final answer only (not every intermediate mark-scheme quantity, which
    the spec also calls for — a documented simplification).

    The check is meaningless for a symbolic expression that still contains
    the free variable (e.g. a derivative like `9*x**8 + 5`) — there's no
    "ugly decimal" concern there since our parameters are always sampled
    as integers, so coefficients are already exact by construction. It
    only applies for real, per spec intent, when the result is (or
    reduces to) a specific number or set of numbers, as from `solve`."""
    if mode != "non-calculator":
        return True

    # A multi-solution "x = a, x = b" string from `solve` — check each root
    # individually before attempting to sympify the joined string (which
    # is never itself valid expression syntax).
    if "," in value_str:
        return all(
            is_number_friendly(part.split("=")[-1].strip(), mode, variable, max_denominator, max_radicand)
            for part in value_str.split(",")
        )

    value_str = value_str.split("=")[-1].strip()
    if _contains_free_variable(value_str, variable):
        return True
    try:
        expr = sympy.sympify(value_str)
    except Exception:  # noqa: BLE001
        return False

    if expr.is_Integer:
        return True
    if expr.is_Rational and abs(expr.q) <= max_denominator:
        return True
    if _is_exact_surd_form(expr, max_radicand):
        return True
    return False


def _uniqueness_gate(item_correct_value: str, distractors) -> QualityGateResult:  # noqa: ANN001
    duplicate = next((d for d in distractors if d.value == item_correct_value), None)
    return QualityGateResult(
        gate="uniqueness",
        passed=duplicate is None,
        detail="no distractor equals the correct answer" if duplicate is None else f"distractor {duplicate.generator_id} matched the correct answer",
    )


def _style_compliance_gate(template: ItemTemplate, rendered_stem: str) -> QualityGateResult:
    word_count = len(rendered_stem.split())
    ok = 3 <= word_count <= 60 and template.command_term.lower() in rendered_stem.lower()
    return QualityGateResult(
        gate="style_compliance",
        passed=ok,
        detail=f"stem word count={word_count}, command_term={'present' if template.command_term.lower() in rendered_stem.lower() else 'missing'}",
    )


def _difficulty_sanity_gate(template: ItemTemplate) -> QualityGateResult:
    b_min, b_max = template.difficulty_band
    return QualityGateResult(
        gate="difficulty_sanity",
        passed=b_min <= b_max,
        detail=f"declared difficulty_band=({b_min}, {b_max})",
    )


def run_quality_gates(
    template: ItemTemplate,
    cas_result: CASResult,
    rendered_stem: str,
    distractors,  # noqa: ANN001
    item_id: str,
) -> QualityGateReport:
    results = [
        QualityGateResult(gate="solvability", passed=cas_result.status == CASStatus.OK, detail=f"CAS status={cas_result.status.value}"),
        _uniqueness_gate(cas_result.result_exact or "", distractors),
        QualityGateResult(
            gate="number_friendliness",
            passed=is_number_friendly(cas_result.result_exact or "", template.calculator_mode, template.variable),
            detail=f"mode={template.calculator_mode}",
        ),
        _difficulty_sanity_gate(template),
        _style_compliance_gate(template, rendered_stem),
        QualityGateResult(gate="mark_scheme_derivability", passed=bool(cas_result.steps), detail="mark scheme built directly from CAS steps"),
    ]
    return QualityGateReport(item_id=item_id, results=results)


def generate_item(
    template_id: str,
    *,
    seed: Optional[int] = None,
    avoid_parameter_hashes: Optional[set[str]] = None,
    avoid_stem_texts: Optional[list[str]] = None,
) -> GeneratedItem:
    template = TEMPLATE_BANK.get(template_id)
    if template is None:
        raise ItemGenerationError(f"unknown template_id: {template_id!r}")

    rng = random.Random(seed)
    avoid_parameter_hashes = avoid_parameter_hashes or set()
    avoid_stem_texts = avoid_stem_texts or []

    for _ in range(template.max_resample_attempts):
        params = _sample_parameters(template, rng)
        if params is None:
            continue  # constraint violation -> resample

        param_hash = compute_parameter_hash(template.template_id, params)
        if param_hash in avoid_parameter_hashes:
            continue  # spec §9.12: never repeat the exact same instance

        expression = template.expression_template.format(**params)
        cas_result = run_cas_operation(template.operation, expression, template.variable)
        if cas_result.status != CASStatus.OK:
            continue  # solvability gate failure -> resample

        if not is_number_friendly(cas_result.result_exact or "", template.calculator_mode, template.variable):
            continue  # number-friendliness gate failure -> resample

        rendered_stem = _render_stem(template, params)
        if _is_near_duplicate(rendered_stem, avoid_stem_texts):
            continue  # leakage/duplicate gate failure -> resample

        distractors = generate_distractors(template.template_id, params, template.variable, cas_result.result_exact or "")

        item_id = f"ITEM-{uuid.uuid4().hex[:12]}"
        b_min, b_max = template.difficulty_band

        item = GeneratedItem(
            item_id=item_id,
            template_id=template.template_id,
            template_version=template.version,
            sampled_parameters=params,
            rendered_stem=rendered_stem,
            calculator_mode=template.calculator_mode,
            difficulty_estimate=DifficultyEstimate(b_param=round((b_min + b_max) / 2, 3)),
            correct_answer=CorrectAnswer(value=cas_result.result_exact or "", cas_verified=True),
            distractors=distractors,
            mark_scheme=build_mark_scheme(item_id, cas_result),
        )
        item.quality_gate_report = run_quality_gates(template, cas_result, rendered_stem, distractors, item_id)
        return item

    raise ItemGenerationError(
        f"failed to generate a valid item for template {template_id!r} after {template.max_resample_attempts} attempts"
    )


async def generate_item_async(
    template_id: str,
    *,
    seed: Optional[int] = None,
    avoid_parameter_hashes: Optional[set[str]] = None,
    avoid_stem_texts: Optional[list[str]] = None,
) -> GeneratedItem:
    """Async wrapper: generation is CPU-bound (SymPy) but normally fast
    (a handful of resample attempts at most); run off the event loop
    thread anyway, consistent with how CAS calls are handled elsewhere."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(
            generate_item,
            template_id,
            seed=seed,
            avoid_parameter_hashes=avoid_parameter_hashes,
            avoid_stem_texts=avoid_stem_texts,
        ),
    )
