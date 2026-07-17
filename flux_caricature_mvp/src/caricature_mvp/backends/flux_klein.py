"""Lazy FLUX.2 Klein editing backend."""

from __future__ import annotations

from typing import Any

from PIL import Image

from ..schemas import GenerationSettings


class FluxKleinBackend:
    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._loaded_key: tuple[str, str, str, bool] | None = None
        self.resolved_device = "unloaded"
        self.resolved_dtype = "unloaded"

    @staticmethod
    def _runtime(settings: GenerationSettings) -> tuple[Any, str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for the flux backend. Install the project dependencies."
            ) from exc
        device = settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_name = settings.dtype
        if dtype_name == "auto":
            if device == "cuda":
                dtype_name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            elif device == "mps":
                dtype_name = "float16"
            else:
                dtype_name = "float32"
        dtype = getattr(torch, dtype_name, None)
        if dtype is None or dtype_name not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be auto, float16, bfloat16, or float32")
        return torch, device, dtype

    def _load(self, settings: GenerationSettings) -> tuple[Any, Any]:
        torch, device, dtype = self._runtime(settings)
        key = (settings.model_id, device, str(dtype), settings.cpu_offload)
        if self._pipeline is not None:
            if key != self._loaded_key:
                raise RuntimeError(
                    "This backend is already loaded with different model/runtime settings; create a new backend instance."
                )
            return torch, self._pipeline
        try:
            from diffusers import Flux2KleinPipeline
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "The installed Diffusers build does not provide Flux2KleinPipeline. "
                "Upgrade Diffusers (pip install -U diffusers) to a FLUX.2-compatible release."
            ) from exc
        try:
            pipeline = Flux2KleinPipeline.from_pretrained(settings.model_id, torch_dtype=dtype)
        except TypeError as exc:
            raise RuntimeError(
                "Diffusers could not construct Flux2KleinPipeline with the supported interface. Upgrade Diffusers."
            ) from exc
        if settings.cpu_offload:
            if device != "cuda":
                raise ValueError("CPU offloading requires device=cuda")
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(device)
        self._pipeline = pipeline
        self._loaded_key = key
        self.resolved_device = device
        self.resolved_dtype = str(dtype).removeprefix("torch.")
        return torch, pipeline

    def generate(
        self,
        input_image: Image.Image,
        prompt: str,
        seed: int,
        settings: GenerationSettings,
    ) -> Image.Image:
        torch, pipeline = self._load(settings)
        generator_device = "cuda" if self.resolved_device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        result = pipeline(
            image=input_image,
            prompt=prompt,
            height=input_image.height,
            width=input_image.width,
            num_inference_steps=settings.num_inference_steps,
            guidance_scale=settings.guidance_scale,
            generator=generator,
        )
        image = result.images[0]
        if not isinstance(image, Image.Image):
            raise RuntimeError("FLUX returned an unsupported output type instead of a PIL image")
        return image.convert("RGB")
