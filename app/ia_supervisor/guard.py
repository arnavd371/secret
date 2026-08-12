"""
Ghostwriting-request guard (spec §11, §2.10): the one hard invariant the
IA Supervisor exists to protect — "never full ghostwriting" — checked
structurally against the student's own request text, before any
coaching response is even attempted. Same posture as
app/orchestrator/signals.py's integrity-risk detector: a deliberately
small, explicit, documented set of regex patterns, not real NLU or
intent classification. A miss here isn't silently fatal — a ghostwritten
paragraph that slips past the input guard would still have to slip past
the Tutor's own structural word-cap and leak checks in tutor_agent.py to
actually reach the student, so this is a layer, not the only layer,
matching the whole codebase's structural-enforcement-over-trust
philosophy.
"""

from __future__ import annotations

import re
from typing import Optional

_CONTENT_NOUNS = r"(?:introduction|conclusion|abstract|analysis|discussion|paragraph|section|essay|research question|methodology|draft|ia|extended essay|ee)"

_GHOSTWRITING_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(rf"\bwrite\s+(?:my|the|a|an)\b(?:\s+\w+){{0,4}}\s+{_CONTENT_NOUNS}\b", re.IGNORECASE),
        "write my/the <IA content> request",
    ),
    (
        re.compile(r"\bwrite\s+(?:it|this|that)\s+for\s+me\b", re.IGNORECASE),
        "write it/this for me",
    ),
    (
        re.compile(r"\b(?:do|finish|complete)\s+(?:my|the)\b(?:\s+\w+){0,3}\s+(?:ia|ee|extended essay|essay|draft|assignment)\b", re.IGNORECASE),
        "do/finish/complete my IA-EE request",
    ),
    (
        re.compile(r"\bgive me\s+(?:a|the)\s+research question\b", re.IGNORECASE),
        "give me a research question",
    ),
    (
        re.compile(r"\bcan you write\b", re.IGNORECASE),
        "can you write ... request",
    ),
]


def detect_ghostwriting_request(raw_input: str) -> Optional[str]:
    """Returns a short description of the matched pattern (useful as
    evidence for the disclosure log and for tests), or None if nothing
    matched — meaning the request looks like legitimate coaching, not a
    request to produce submittable content."""
    for pattern, description in _GHOSTWRITING_PATTERNS:
        if pattern.search(raw_input):
            return description
    return None
