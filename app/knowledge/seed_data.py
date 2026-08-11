"""
A small, real seed of IB Mathematics: Analysis & Approaches curriculum
content — not the full syllabus (spec §16.7 rightly calls full-syllabus
authoring a large, curriculum-team-heavy effort on its own), but enough
real, correctly-sourced entries to exercise retrieval end to end and
ground a handful of common Tutor responses in an actual citation rather
than an empty placeholder.
"""

from __future__ import annotations

from app.knowledge.schemas import (
    DocType,
    FormulaBookletEntry,
    KnowledgeChunk,
    LearningObjective,
    WorkedExample,
    WorkedExampleStep,
)

_LEARNING_OBJECTIVES = [
    LearningObjective(
        id="LO-AA-5.6.1",
        subtopic_id="calculus.differentiation.chain_rule",
        text="Differentiate composite functions using the chain rule.",
        level="SL",
        source="IB DP Mathematics: analysis and approaches guide, section 5.6",
    ),
    LearningObjective(
        id="LO-AA-2.5.1",
        subtopic_id="algebra.quadratics.solving",
        text=(
            "Solve quadratic equations and quadratic inequalities using the quadratic formula, and interpret "
            "the discriminant to determine the number and nature of the roots."
        ),
        level="SL",
        source="IB DP Mathematics: analysis and approaches guide, section 2.5",
    ),
    LearningObjective(
        id="LO-AA-5.9.1",
        subtopic_id="calculus.integration.reverse_power_rule",
        text="Integrate using the reverse of differentiation, including the addition of a constant of integration.",
        level="SL",
        source="IB DP Mathematics: analysis and approaches guide, section 5.9",
    ),
]

_FORMULA_BOOKLET_ENTRIES = [
    FormulaBookletEntry(
        id="FB-AA-5.7",
        section="Calculus",
        level="SL",
        latex=r"\frac{d}{dx}(uv) = u'v + uv'",
        name="Product rule",
        applies_to_subtopics=["calculus.differentiation.product_rule"],
    ),
    FormulaBookletEntry(
        id="FB-AA-5.8",
        section="Calculus",
        level="SL",
        latex=r"\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}",
        name="Chain rule",
        applies_to_subtopics=["calculus.differentiation.chain_rule"],
    ),
    FormulaBookletEntry(
        id="FB-AA-5.2",
        section="Calculus",
        level="SL",
        latex=r"\frac{d}{dx}(\sin x) = \cos x, \quad \frac{d}{dx}(\cos x) = -\sin x",
        name="Derivatives of sine and cosine",
        applies_to_subtopics=["calculus.differentiation.trig_functions"],
    ),
    FormulaBookletEntry(
        id="FB-AA-2.5",
        section="Algebra",
        level="SL",
        latex=r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
        name="Quadratic formula",
        applies_to_subtopics=["algebra.quadratics.solving"],
    ),
    FormulaBookletEntry(
        id="FB-AA-3.6",
        section="Geometry and Trigonometry",
        level="SL",
        latex=r"\sin(A + B) = \sin A \cos B + \cos A \sin B",
        name="Compound angle identity",
        applies_to_subtopics=["trigonometry.identities.compound_angle"],
    ),
]

_WORKED_EXAMPLES = [
    WorkedExample(
        id="WE-calc-chainrule-001",
        subtopic_id="calculus.differentiation.chain_rule",
        stem_latex=r"Find \frac{dy}{dx} where y = (3x^2 + 1)^5.",
        steps=[
            WorkedExampleStep(step=1, latex=r"\text{let } u = 3x^2 + 1", explanation="identify the inner function"),
            WorkedExampleStep(step=2, latex=r"y = u^5,\ \frac{dy}{du} = 5u^4", explanation="differentiate the outer function"),
            WorkedExampleStep(step=3, latex=r"\frac{du}{dx} = 6x", explanation="differentiate the inner function"),
            WorkedExampleStep(
                step=4,
                latex=r"\frac{dy}{dx} = 5u^4 \cdot 6x = 30x(3x^2+1)^4",
                explanation="apply the chain rule, substitute back",
            ),
        ],
    ),
]

# Keyword aliases per subtopic, to improve lexical recall over the raw
# spec/formula text (a real student rarely types the syllabus_ref).
_KEYWORDS_BY_SUBTOPIC: dict[str, list[str]] = {
    "calculus.differentiation.chain_rule": ["chain rule", "composite function", "inner function", "outer function", "u substitution"],
    "calculus.differentiation.product_rule": ["product rule", "differentiate a product", "u v prime", "multiply two functions"],
    "calculus.differentiation.trig_functions": ["derivative of sin", "derivative of cos", "differentiate sine", "differentiate cosine", "trig derivative"],
    "algebra.quadratics.solving": ["quadratic formula", "solve quadratic", "discriminant", "roots of a quadratic"],
    "calculus.integration.reverse_power_rule": ["integrate", "integration", "antiderivative", "reverse power rule", "constant of integration"],
    "trigonometry.identities.compound_angle": ["compound angle", "sin a+b", "angle sum identity", "trig identity"],
}


def _chunks_from_learning_objectives() -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk(
            chunk_id=lo.id,
            doc_type=DocType.LEARNING_OBJECTIVE,
            subtopic_id=lo.subtopic_id,
            citation=lo.source,
            text=lo.text,
            keywords=_KEYWORDS_BY_SUBTOPIC.get(lo.subtopic_id, []),
            authority_tier=1.0,
        )
        for lo in _LEARNING_OBJECTIVES
    ]


def _chunks_from_formula_booklet() -> list[KnowledgeChunk]:
    chunks = []
    for fb in _FORMULA_BOOKLET_ENTRIES:
        subtopic_id = fb.applies_to_subtopics[0] if fb.applies_to_subtopics else "unclassified"
        chunks.append(
            KnowledgeChunk(
                chunk_id=fb.id,
                doc_type=DocType.FORMULA_BOOKLET_ENTRY,
                subtopic_id=subtopic_id,
                citation=f"Formula booklet, {fb.section}: {fb.name}",
                text=f"{fb.name}: {fb.latex}",
                keywords=[fb.name.lower(), *_KEYWORDS_BY_SUBTOPIC.get(subtopic_id, [])],
                authority_tier=1.0,
            )
        )
    return chunks


def _chunks_from_worked_examples() -> list[KnowledgeChunk]:
    chunks = []
    for we in _WORKED_EXAMPLES:
        step_text = " ".join(f"{s.explanation}: {s.latex}" for s in we.steps)
        chunks.append(
            KnowledgeChunk(
                chunk_id=we.id,
                doc_type=DocType.WORKED_EXAMPLE,
                subtopic_id=we.subtopic_id,
                citation=f"Worked example {we.id}",
                text=f"{we.stem_latex} {step_text}",
                keywords=_KEYWORDS_BY_SUBTOPIC.get(we.subtopic_id, []),
                authority_tier=0.7,
            )
        )
    return chunks


def load_seed_chunks() -> list[KnowledgeChunk]:
    return [
        *_chunks_from_learning_objectives(),
        *_chunks_from_formula_booklet(),
        *_chunks_from_worked_examples(),
    ]
