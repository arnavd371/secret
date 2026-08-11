"""
Examiner-comment generation (spec §10.7): "Comments are generated from a
grounded template + LLM fill approach, never free-floating LLM
commentary... The LLM's role is restricted to natural-language phrasing
of the above structured facts."

Phase 4 implements the grounded-template half for real and skips the LLM
phrasing pass entirely — the comment below states nothing beyond what
`MarkResult`/`MarkScheme` already computed, so there's no unsupported
claim an LLM phrasing pass could introduce, and no LLM call needed to get
a safe, grounded comment. A warmer natural-language rephrasing of this
same content is a reasonable later enhancement, not a correctness gap.
"""

from __future__ import annotations

from app.examiner.models import MarkResult
from app.questions.models import MarkScheme


def generate_examiner_comment(mark_result: MarkResult, mark_scheme: MarkScheme) -> str:
    nodes_by_id = {node.id: node for node in mark_scheme.nodes}

    lines = [f"Total: {mark_result.total_awarded}/{mark_result.total_available} marks."]

    for award in mark_result.breakdown:
        node = nodes_by_id.get(award.node_id)
        node_text = node.text if node else award.node_id
        if award.marks_awarded == award.marks_available:
            lines.append(f"{award.node_id} awarded: {node_text}.")
        else:
            lines.append(f"{award.node_id} not awarded: {node_text} — {award.reason}.")

    if "unsupported_correct_answer" in mark_result.flags:
        lines.append(
            "Your final answer is correct, but the working shown doesn't fully support it — show each "
            "step so method marks can be awarded."
        )

    if mark_result.first_error_step_index is not None:
        lines.append(
            f"The first place this diverges from a fully correct solution is around step "
            f"{mark_result.first_error_step_index + 1} of your working."
        )

    return " ".join(lines)
