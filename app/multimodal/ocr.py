"""
Math OCR call (spec §3.2, §14.2): the pipeline's one LLM call, routed
through the `math_ocr` capability so the concrete vision model is
configured in exactly one place (app/llm/router_config.py), same
discipline as every other capability in this codebase.

The system prompt is deliberately transcription-only: the model's job is
to read what's on the page, not to grade it, fix it, or solve it. Mixing
"transcribe" and "evaluate" into one call would make a bad transcription
indistinguishable from a bad grade — keeping them separate lets Phase 4's
already-real grader operate on the transcription exactly as it would on
typed text.

`ModelUnavailableError` is allowed to propagate out of `transcribe_image`
uncaught, same convention as the rest of the router-calling code
(app/agents/tutor_agent.py, app/verifier/critic.py): the caller decides
how to degrade, this module doesn't hide the failure.
"""

from __future__ import annotations

from app.llm.client import ImageInput, ModelRouter
from app.multimodal.models import OCRResult

_SYSTEM_PROMPT = """You are a precise math transcription assistant. Your \
only job is to transcribe the mathematical work shown in the image \
exactly as written.

Rules:
- Transcribe all mathematical expressions using LaTeX notation.
- Transcribe any surrounding words exactly as written.
- Do NOT solve, simplify, correct, or complete the work. Transcribe \
errors exactly as the student wrote them.
- If a part of the image is illegible, write [illegible] for that part \
instead of guessing what it might say.
- Preserve the line-by-line structure of the work as closely as possible.
- Output only the transcription, no commentary, no markdown code fences.
"""

_USER_PROMPT = "Transcribe the mathematical work shown in this image."


async def transcribe_image(router: ModelRouter, processed_image_bytes: bytes) -> OCRResult:
    result = await router.call(
        capability="math_ocr",
        system=_SYSTEM_PROMPT,
        user=_USER_PROMPT,
        images=[ImageInput(data=processed_image_bytes, media_type="image/png")],
    )
    return OCRResult(raw_text=result.text, model=result.model)
