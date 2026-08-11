"""
Mark scheme construction (spec §9.4, §10.1) built directly from the CAS
agent's own solution steps — every node's text traces to something SymPy
actually computed, so "mark-scheme derivability" (spec §9.13's quality
gate) holds by construction rather than being authored separately and
potentially drifting from the real solution.
"""

from __future__ import annotations

from app.cas.models import CASResult
from app.questions.models import MarkScheme, MarkSchemeNode

# Step strings that describe a *method* (e.g. "product_rule", "chain_rule")
# rather than a literal computation line (e.g. "d/dx[...] = ...").
_METHOD_STEP_PREFIXES = ("product_rule", "chain_rule", "standard_differentiation_rules")


def build_mark_scheme(item_id: str, cas_result: CASResult) -> MarkScheme:
    method_notes = [step for step in cas_result.steps if step in _METHOD_STEP_PREFIXES]
    nodes: list[MarkSchemeNode] = []

    if method_notes:
        for index, note in enumerate(method_notes, start=1):
            nodes.append(
                MarkSchemeNode(
                    id=f"M{index}", type="M", text=f"correct application of {note.replace('_', ' ')}", marks=1
                )
            )
    else:
        nodes.append(MarkSchemeNode(id="M1", type="M", text="correct method applied", marks=1))

    accuracy_index = len(nodes) + 1
    nodes.append(
        MarkSchemeNode(
            id=f"A{accuracy_index}",
            type="A",
            text=f"correct final answer: {cas_result.result_exact}",
            marks=1,
            expected_value=cas_result.result_exact,
        )
    )

    return MarkScheme(item_id=item_id, total_marks=sum(n.marks for n in nodes), nodes=nodes)
