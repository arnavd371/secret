"""
Real PIL-based image preprocessing (spec §3.2), run on every intake-
accepted image before it reaches the vision model. Three deterministic
steps, in order:

  1. Grayscale conversion (`Image.convert("L")`).
  2. Contrast normalization (`ImageOps.autocontrast`), which stretches
     the histogram to use the full 0-255 range — corrects for the flat,
     washed-out contrast a phone camera produces under uneven lighting.
  3. Binarization via a real Otsu threshold (not a fixed cutoff): the
     threshold that minimizes intra-class pixel-value variance is
     computed from the image's own histogram, so it adapts to how dark
     or light a given photo actually is.

Followed by a resize pass that keeps the image inside an OCR-friendly
resolution band: downscale if it's needlessly huge (bounds request
latency and cost), upscale if it's small enough that fine strokes would
be lost.

Documented tradeoff: hard binarization is the classical preprocessing
step for legacy OCR engines (Tesseract and similar), which threshold
pixels before running edge/stroke detection. A vision-LLM-based OCR
step (this pipeline's actual approach, see ocr.py) reads grayscale
images natively and can use anti-aliasing information that binarization
throws away. This pipeline still binarizes, because spec §3.2 calls for
it as an explicit preprocessing capability and because it's a cheap,
deterministic, testable step — but a production system tuned
specifically around a vision-LLM backend might reasonably drop this
step, or make it conditional on a low-contrast heuristic. Not attempted
here; flagged rather than silently decided.
"""

from __future__ import annotations

import io

from PIL import Image, ImageOps

from app.multimodal.models import PreprocessResult

RESIZE_MAX_DIMENSION_PX = 2000
RESIZE_MIN_DIMENSION_PX = 600


def _otsu_threshold(histogram: list[int]) -> int:
    """Standard Otsu's method: the threshold in [0, 255] that maximizes
    between-class variance (equivalently minimizes within-class
    variance) of the pixel histogram. Pure arithmetic, no PIL/numpy
    dependency beyond the histogram itself."""
    total = sum(histogram)
    if total == 0:
        return 128

    sum_total = sum(i * count for i, count in enumerate(histogram))
    sum_background = 0.0
    weight_background = 0
    best_variance = -1.0
    # Track every threshold that ties for the maximum between-class
    # variance (a flat plateau is common whenever there's a gap with no
    # pixels in it, e.g. a genuinely bimodal image) and land in the
    # middle of that plateau, rather than silently taking whichever end
    # the loop happens to visit first.
    best_thresholds: list[int] = [0]

    for i in range(256):
        weight_background += histogram[i]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += i * histogram[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        between_class_variance = (
            weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        )
        if between_class_variance > best_variance:
            best_variance = between_class_variance
            best_thresholds = [i]
        elif between_class_variance == best_variance:
            best_thresholds.append(i)

    return (best_thresholds[0] + best_thresholds[-1]) // 2


def preprocess_image(image_bytes: bytes) -> PreprocessResult:
    with Image.open(io.BytesIO(image_bytes)) as original:
        original.load()
        was_grayscale = original.mode == "L"
        gray = original.convert("L")

    contrasted = ImageOps.autocontrast(gray, cutoff=1)

    threshold = _otsu_threshold(contrasted.histogram())
    binarized = contrasted.point(lambda pixel, t=threshold: 255 if pixel > t else 0)

    width, height = binarized.size
    longest = max(width, height)
    shortest = min(width, height)
    working = binarized
    resized = False

    if longest > RESIZE_MAX_DIMENSION_PX:
        scale = RESIZE_MAX_DIMENSION_PX / longest
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        working = working.resize(new_size, Image.LANCZOS)
        resized = True
    elif shortest < RESIZE_MIN_DIMENSION_PX:
        scale = RESIZE_MIN_DIMENSION_PX / shortest
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        working = working.resize(new_size, Image.LANCZOS)
        resized = True

    buffer = io.BytesIO()
    working.save(buffer, format="PNG")

    return PreprocessResult(
        processed_image_bytes=buffer.getvalue(),
        width_px=working.width,
        height_px=working.height,
        grayscale_applied=not was_grayscale,
        contrast_enhanced=True,
        binarized=True,
        resized=resized,
    )
