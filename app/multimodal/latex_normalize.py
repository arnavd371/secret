"""
LaTeX normalization (spec §3.2): cleans up the raw OCR transcription
before anything tries to parse it as math. Deterministic string
transforms only, no model call.

  - strips markdown code fences a vision model sometimes wraps output in
  - strips inline/display math delimiters (\\(...\\), \\[...\\], $...$,
    $$...$$), keeping their content
  - resolves a handful of common command aliases (\\dfrac/\\tfrac -> \\frac)
    so later stages only need to handle one spelling
  - collapses blank-line runs and trailing whitespace, while preserving
    genuine line breaks — those line breaks are exactly what Phase 4's
    step segmentation (app/examiner/segmentation.py) splits on, so this
    step must not flatten multi-step work into one line
"""

from __future__ import annotations

import re

from app.multimodal.models import NormalizedTranscription

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?|```\s*$", re.MULTILINE)

_MATH_DELIMITER_PATTERNS = [
    re.compile(r"\\\((.*?)\\\)", re.DOTALL),
    re.compile(r"\\\[(.*?)\\\]", re.DOTALL),
    re.compile(r"\$\$(.*?)\$\$", re.DOTALL),
    re.compile(r"\$(.*?)\$", re.DOTALL),
]

_COMMAND_ALIASES = {
    r"\\dfrac": r"\\frac",
    r"\\tfrac": r"\\frac",
    r"\\cfrac": r"\\frac",
}

_LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+")


def normalize_transcription(raw_text: str) -> NormalizedTranscription:
    text = _CODE_FENCE_RE.sub("", raw_text)

    for pattern in _MATH_DELIMITER_PATTERNS:
        text = pattern.sub(lambda m: m.group(1), text)

    for alias, canonical in _COMMAND_ALIASES.items():
        text = re.sub(alias, canonical, text)

    lines = [line.rstrip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = line.strip() == ""
        if is_blank and previous_blank:
            continue
        cleaned_lines.append(line)
        previous_blank = is_blank

    normalized = "\n".join(cleaned_lines).strip("\n")
    normalized = normalized.strip()

    return NormalizedTranscription(
        normalized_text=normalized,
        latex_command_count=len(_LATEX_COMMAND_RE.findall(normalized)),
    )
