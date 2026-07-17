"""Small environment-aware configuration helpers."""

from __future__ import annotations

import os

from .schemas import GenerationSettings


def settings_from_environment(**overrides: object) -> GenerationSettings:
    values: dict[str, object] = {
        "model_id": os.getenv("CARICATURE_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B"),
        "device": os.getenv("CARICATURE_DEVICE", "auto"),
        "dtype": os.getenv("CARICATURE_DTYPE", "auto"),
        "max_side": int(os.getenv("CARICATURE_MAX_SIDE", "1024")),
        "cpu_offload": os.getenv("CARICATURE_CPU_OFFLOAD", "0").lower() in {"1", "true", "yes"},
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    return GenerationSettings(**values)
