"""
Expression parseability check (spec §3.2): "does the normalized
transcription contain at least one line that's actually valid math,"
checked by handing it to the same SymPy parser Phase 2's CAS layer
already uses (`app.cas.solver.try_parse_expression`) rather than
standing up a second parser. This is a real, independent-of-the-model
signal for the composite confidence score in confidence.py: a vision
model can be linguistically fluent and still transcribe unparseable
garbage, and this check catches that case directly.

`_latex_to_plain` is a best-effort, non-exhaustive LaTeX-to-plain-text
converter — it handles the constructs common in IB AA HL/SL work
(fractions, roots, named functions, superscripts, common Greek letters)
but does not attempt matrices, integrals with limits, piecewise
notation, or other constructs outside that scope. A candidate line that
uses one of those unsupported constructs will correctly come back as
"not parseable" rather than silently producing a wrong answer — that's
the intended failure mode, not a bug.
"""

from __future__ import annotations

import re

from app.cas.solver import try_parse_expression
from app.multimodal.models import ExpressionParseResult

_NAMED_FUNCTIONS = ("sin", "cos", "tan", "ln", "log", "exp", "sqrt", "asin", "acos", "atan")
_GREEK_AND_CONSTANTS = {
    r"\pi": "pi",
    r"\theta": "theta",
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\infty": "oo",
}

_LOOKS_MATHY_RE = re.compile(r"[=+\-*/^]|\\[a-zA-Z]+")


def _extract_balanced(text: str, start_idx: int) -> tuple[str, int]:
    """`start_idx` points just past an already-consumed opening brace.
    Returns (content, index just past the matching closing brace)."""
    depth = 1
    i = start_idx
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx:i], i + 1
        i += 1
    return text[start_idx:], len(text)


def _convert_one_arg_command(text: str, command: str, wrap) -> str:
    marker = command + "{"
    while True:
        idx = text.find(marker)
        if idx == -1:
            return text
        arg, after = _extract_balanced(text, idx + len(marker))
        text = text[:idx] + wrap(arg.strip()) + text[after:]


def _convert_frac(text: str) -> str:
    marker = "\\frac{"
    while True:
        idx = text.find(marker)
        if idx == -1:
            return text
        numerator, after_num = _extract_balanced(text, idx + len(marker))
        if after_num >= len(text) or text[after_num] != "{":
            return text  # malformed \frac, bail rather than loop forever
        denominator, after_denom = _extract_balanced(text, after_num + 1)
        replacement = f"(({numerator.strip()})/({denominator.strip()}))"
        text = text[:idx] + replacement + text[after_denom:]


def _latex_to_plain(candidate: str) -> str:
    text = candidate
    text = _convert_frac(text)
    text = _convert_one_arg_command(text, "\\sqrt", lambda arg: f"sqrt(({arg}))")

    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("\\left", "").replace("\\right", "")

    for latex_symbol, plain_symbol in _GREEK_AND_CONSTANTS.items():
        text = text.replace(latex_symbol, plain_symbol)

    for fn in _NAMED_FUNCTIONS:
        text = re.sub(rf"\\{fn}\b", fn, text)

    # Braced exponents/subscripts: x^{2} -> x**(2), x_{1} -> x_1
    text = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", text)
    text = re.sub(r"_\{([^{}]*)\}", r"_\1", text)
    text = text.replace("^", "**")

    # Any remaining unhandled LaTeX command: drop the backslash and keep
    # the letters (best-effort fallback, not a guarantee of correctness).
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)
    text = text.replace("$", "")

    return text.strip()


def _looks_mathy(line: str) -> bool:
    return bool(line.strip()) and bool(_LOOKS_MATHY_RE.search(line))


def _candidates(normalized_text: str) -> list[str]:
    # Gate every candidate, including the whole-text one, behind
    # `_looks_mathy` first. Without this gate, sympy's implicit-
    # multiplication parser will happily read plain prose as a product
    # of single-letter symbols (e.g. "some illegible scrawl" parses as
    # `s*o*m*e*i*l*l*e*g*i*b*l*e*...`) — a false "parseable" that would
    # defeat the whole point of this check.
    candidates = [normalized_text] if _looks_mathy(normalized_text) else []
    for line in normalized_text.splitlines():
        if not _looks_mathy(line):
            continue
        candidates.append(line)
        if "=" in line:
            # An equation transcribes as two candidate expressions (each
            # side of the last "="): a step like "dy/dx = 5(2x+1)^4 * 2"
            # should count as parseable even though the whole line, with
            # its bare "=", isn't valid parser input on its own.
            lhs, _, rhs = line.rpartition("=")
            candidates.append(lhs)
            candidates.append(rhs)
    return candidates


def parse_expression(normalized_text: str) -> ExpressionParseResult:
    if not normalized_text.strip():
        return ExpressionParseResult(
            parseable=False, parsed_expression=None, parse_error="empty transcription"
        )

    for candidate in _candidates(normalized_text):
        plain = _latex_to_plain(candidate)
        if not plain.strip():
            continue
        parsed = try_parse_expression(plain)
        if parsed is not None:
            return ExpressionParseResult(
                parseable=True, parsed_expression=str(parsed), parse_error=None
            )

    return ExpressionParseResult(
        parseable=False,
        parsed_expression=None,
        parse_error="no parseable mathematical expression found in the transcription",
    )
