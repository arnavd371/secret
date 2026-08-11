"""
Misconception catalog (spec §8.2), scoped to the same named errors Phase
3's distractor generators (app/questions/distractors.py) already encode
as deliberate wrong answers. Reusing the exact same IDs means a
misconception diagnosed here and a misconception a generated item's
distractor is designed to bait are traceably the same catalog entry, not
two parallel taxonomies that happen to look similar.

Not modeled here (later-phase non-goal): the full spec §8.2 catalog
covers many more error patterns across the AA HL/SL syllabus. This
build's catalog only covers what app/diagnostician/detectors.py can
actually detect for real plus what Phase 3 already generates distractors
for — a small, honest, real subset rather than a large catalog with no
detection behind most of it.
"""

from __future__ import annotations

MISCONCEPTION_CATALOG: dict[str, str] = {
    "MISC-CALC-010": "Differentiates a product using f'(x)g'(x) instead of the product rule.",
    "MISC-CALC-014": "Applies the power rule to the outer function in a composite but forgets to multiply by the derivative of the inner function.",
    "MISC-ALG-003": "Sign error distributing a negative across a bracket (e.g. using +b instead of -b in the quadratic formula).",
    "MISC-CALC-GEN-DROPPED-TERM": "Drops one additive term entirely when differentiating a sum.",
}


def describe(misconception_id: str) -> str:
    return MISCONCEPTION_CATALOG.get(misconception_id, "unknown misconception")
