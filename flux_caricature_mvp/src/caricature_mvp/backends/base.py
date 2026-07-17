"""Image editing backend protocol."""

from typing import Protocol

from PIL import Image

from ..schemas import GenerationSettings


class ImageEditBackend(Protocol):
    def generate(
        self,
        input_image: Image.Image,
        prompt: str,
        seed: int,
        settings: GenerationSettings,
    ) -> Image.Image: ...
