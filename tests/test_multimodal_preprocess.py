"""
Real PIL preprocessing assertions — pixel-level checks against synthetic
images, not just "did it run without throwing."
"""

import io

from PIL import Image

from app.multimodal.preprocess import (
    RESIZE_MAX_DIMENSION_PX,
    RESIZE_MIN_DIMENSION_PX,
    _otsu_threshold,
    preprocess_image,
)


def _rgb_png_bytes(width: int, height: int, color=(180, 90, 40)) -> bytes:
    image = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _half_and_half_png_bytes(width: int, height: int) -> bytes:
    """A synthetic image with a genuinely bimodal histogram: left half
    dark, right half light, so Otsu's threshold has real separation to
    find rather than a flat, degenerate histogram."""
    image = Image.new("L", (width, height), color=0)
    for x in range(width // 2, width):
        for y in range(height):
            image.putpixel((x, y), 255)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_otsu_threshold_separates_a_clean_bimodal_histogram():
    histogram = [0] * 256
    histogram[20] = 1000  # dark cluster
    histogram[230] = 1000  # light cluster
    threshold = _otsu_threshold(histogram)
    assert 20 < threshold < 230


def test_otsu_threshold_handles_empty_histogram_without_crashing():
    assert _otsu_threshold([0] * 256) == 128


def test_preprocess_converts_color_image_to_grayscale_source():
    result = preprocess_image(_rgb_png_bytes(900, 700))
    assert result.grayscale_applied is True
    with Image.open(io.BytesIO(result.processed_image_bytes)) as processed:
        # binarized output is still single-channel
        assert processed.mode == "L"


def test_preprocess_output_is_binary_valued():
    result = preprocess_image(_half_and_half_png_bytes(900, 700))
    assert result.binarized is True
    with Image.open(io.BytesIO(result.processed_image_bytes)) as processed:
        pixel_values = set(processed.tobytes())
    assert pixel_values.issubset({0, 255})


def test_preprocess_downscales_an_oversized_image():
    result = preprocess_image(_rgb_png_bytes(4000, 3000))
    assert result.resized is True
    assert max(result.width_px, result.height_px) <= RESIZE_MAX_DIMENSION_PX
    # aspect ratio preserved (4000x3000 = 4:3)
    assert abs(result.width_px / result.height_px - 4000 / 3000) < 0.01


def test_preprocess_upscales_an_undersized_image():
    result = preprocess_image(_rgb_png_bytes(150, 120))
    assert result.resized is True
    assert min(result.width_px, result.height_px) >= RESIZE_MIN_DIMENSION_PX


def test_preprocess_leaves_a_well_sized_image_unresized():
    result = preprocess_image(_rgb_png_bytes(1200, 900))
    assert result.resized is False
    assert result.width_px == 1200
    assert result.height_px == 900


def test_preprocess_output_is_valid_png_bytes():
    result = preprocess_image(_rgb_png_bytes(500, 400))
    with Image.open(io.BytesIO(result.processed_image_bytes)) as processed:
        assert processed.format == "PNG"
