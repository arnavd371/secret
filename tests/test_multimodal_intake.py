"""
Real checks against the intake gate — synthetic images generated with
PIL at test time, no fixture files needed.
"""

import io

from PIL import Image

from app.multimodal.intake import (
    MAX_DIMENSION_PX,
    MAX_SIZE_BYTES,
    MIN_DIMENSION_PX,
    validate_intake,
)
from app.multimodal.models import ImageFormat, IntakeRejectionReason


def _png_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), color=(200, 200, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_valid_png_is_accepted():
    result = validate_intake(_png_bytes(400, 300))
    assert result.accepted is True
    assert result.format == ImageFormat.PNG
    assert result.width_px == 400
    assert result.height_px == 300
    assert result.rejection_reason is None


def test_valid_jpeg_is_accepted():
    result = validate_intake(_jpeg_bytes(500, 500))
    assert result.accepted is True
    assert result.format == ImageFormat.JPEG


def test_corrupt_bytes_are_rejected():
    result = validate_intake(b"not an image, just some random bytes")
    assert result.accepted is False
    assert result.rejection_reason == IntakeRejectionReason.CORRUPT_IMAGE


def test_unsupported_format_is_rejected():
    image = Image.new("RGB", (400, 400), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="BMP")
    result = validate_intake(buffer.getvalue())
    assert result.accepted is False
    assert result.rejection_reason == IntakeRejectionReason.UNSUPPORTED_FORMAT


def test_too_small_dimensions_are_rejected():
    result = validate_intake(_png_bytes(MIN_DIMENSION_PX - 1, MIN_DIMENSION_PX - 1))
    assert result.accepted is False
    assert result.rejection_reason == IntakeRejectionReason.DIMENSIONS_TOO_SMALL


def test_too_large_dimensions_are_rejected():
    result = validate_intake(_png_bytes(MAX_DIMENSION_PX + 1, 400))
    assert result.accepted is False
    assert result.rejection_reason == IntakeRejectionReason.DIMENSIONS_TOO_LARGE


def test_oversized_byte_payload_is_rejected_before_decoding():
    oversized = b"\x89PNG\r\n\x1a\n" + b"0" * (MAX_SIZE_BYTES + 1)
    result = validate_intake(oversized)
    assert result.accepted is False
    assert result.rejection_reason == IntakeRejectionReason.TOO_LARGE_BYTES
    assert result.size_bytes == len(oversized)


def test_boundary_dimensions_exactly_at_minimum_are_accepted():
    result = validate_intake(_png_bytes(MIN_DIMENSION_PX, MIN_DIMENSION_PX))
    assert result.accepted is True
