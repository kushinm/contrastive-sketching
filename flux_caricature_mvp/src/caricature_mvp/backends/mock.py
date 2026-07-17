"""Deterministic CPU-only backend used for tests and demos."""

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from ..schemas import GenerationSettings


class MockBackend:
    def generate(
        self,
        input_image: Image.Image,
        prompt: str,
        seed: int,
        settings: GenerationSettings,
    ) -> Image.Image:
        del prompt, settings
        gray = ImageOps.grayscale(input_image)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageEnhance.Contrast(edges).enhance(2.5)
        result = ImageOps.invert(edges).convert("RGB")
        draw = ImageDraw.Draw(result)
        label = f"MOCK seed {seed}"
        box = draw.textbbox((0, 0), label)
        draw.rectangle((8, 8, box[2] + 16, box[3] + 16), fill="white", outline="black")
        draw.text((12, 12), label, fill="black")
        return result
