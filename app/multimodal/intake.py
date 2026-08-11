"""
Image intake gate (spec §3.2): the first, cheapest checkpoint in the
multimodal pipeline. Runs before any model call or heavy processing, so a
malformed upload fails fast and for free.

Real checks, no LLM involved:
  - format: only PNG/JPEG accepted (what a phone camera or screenshot
    tool actually produces; HEIC and PDF are explicitly out of scope,
    see the TODO below)
  - byte size: reject anything absurdly large before it's even decoded
  - decodability: PIL must be able to open and identify it, or it's
    treated as corrupt
  - pixel dimensions: reject images too small to plausibly contain
    legible handwriting, and images large enough to be a decompression-
    bomb risk rather than a real photo

TODO (spec §3.2, not built): multi-page PDF intake, HEIC support (iOS's
native camera format — would need pillow-heif, an extra dependency this
build doesn't take on), and virus/malware scanning of uploaded bytes.
"""

from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError

from app.multimodal.models import ImageFormat, IntakeRejectionReason, IntakeResult

MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_DIMENSION_PX = 100
MAX_DIMENSION_PX = 6000

_PIL_FORMAT_TO_ENUM = {
    "PNG": ImageFormat.PNG,
    "JPEG": ImageFormat.JPEG,
}


def validate_intake(image_bytes: bytes) -> IntakeResult:
    size_bytes = len(image_bytes)

    if size_bytes > MAX_SIZE_BYTES:
        return IntakeResult(
            accepted=False,
            size_bytes=size_bytes,
            rejection_reason=IntakeRejectionReason.TOO_LARGE_BYTES,
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return IntakeResult(
            accepted=False,
            size_bytes=size_bytes,
            rejection_reason=IntakeRejectionReason.CORRUPT_IMAGE,
        )

    # verify() leaves the file object unusable for further reads, so
    # re-open to actually inspect format/dimensions.
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            pil_format = image.format
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError):
        return IntakeResult(
            accepted=False,
            size_bytes=size_bytes,
            rejection_reason=IntakeRejectionReason.CORRUPT_IMAGE,
        )

    image_format = _PIL_FORMAT_TO_ENUM.get(pil_format or "")
    if image_format is None:
        return IntakeResult(
            accepted=False,
            size_bytes=size_bytes,
            rejection_reason=IntakeRejectionReason.UNSUPPORTED_FORMAT,
        )

    if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
        return IntakeResult(
            accepted=False,
            format=image_format,
            width_px=width,
            height_px=height,
            size_bytes=size_bytes,
            rejection_reason=IntakeRejectionReason.DIMENSIONS_TOO_SMALL,
        )

    if width > MAX_DIMENSION_PX or height > MAX_DIMENSION_PX:
        return IntakeResult(
            accepted=False,
            format=image_format,
            width_px=width,
            height_px=height,
            size_bytes=size_bytes,
            rejection_reason=IntakeRejectionReason.DIMENSIONS_TOO_LARGE,
        )

    return IntakeResult(
        accepted=True,
        format=image_format,
        width_px=width,
        height_px=height,
        size_bytes=size_bytes,
        rejection_reason=None,
    )
