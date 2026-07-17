"""Input image validation and non-cropping normalization."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image, ImageOps

SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP"}
MIN_DIMENSION = 32


def normalize_image(
    source: str | Path | Image.Image,
    max_side: int = 1024,
    alignment: int = 16,
) -> Image.Image:
    if isinstance(source, Image.Image):
        image = source.copy()
        image_format = source.format
    else:
        path = Path(source)
        try:
            image = Image.open(path)
            image_format = image.format
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read input image: {path}") from exc
    if image_format and image_format.upper() not in SUPPORTED_FORMATS:
        raise ValueError("Input must be PNG, JPEG, or WebP")
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise ValueError(f"Input dimensions must each be at least {MIN_DIMENSION} pixels")
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        resized = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(resized, Image.Resampling.LANCZOS)
    if alignment > 1:
        padded_width = ((image.width + alignment - 1) // alignment) * alignment
        padded_height = ((image.height + alignment - 1) // alignment) * alignment
        if (padded_width, padded_height) != image.size:
            canvas = Image.new("RGB", (padded_width, padded_height), "white")
            offset = ((padded_width - image.width) // 2, (padded_height - image.height) // 2)
            canvas.paste(image, offset)
            image = canvas
    return image


def save_normalized_input(image: Image.Image, destination: str | Path) -> str:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")
    return sha256_image_file(destination)


def sha256_image_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
