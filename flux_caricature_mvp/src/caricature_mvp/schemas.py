"""Typed application models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class CaricatureMode(str, Enum):
    PART_EMPHASIS = "part_emphasis"
    CATEGORY_SELECTIVITY = "category_selectivity"
    ICONIC_SIMPLIFICATION = "iconic_simplification"
    FUNCTIONAL_SELECTIVITY = "functional_selectivity"


class GenerationSpec(BaseModel):
    output_index: int = Field(ge=1, le=10)
    mode: CaricatureMode
    variant: str
    strength: str
    instruction: str
    seed: int = Field(ge=0)


class GenerationSettings(BaseModel):
    model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    device: str = "auto"
    dtype: str = "auto"
    max_side: int = Field(default=1024, ge=64)
    cpu_offload: bool = False
    num_inference_steps: int = Field(default=4, ge=1)
    guidance_scale: float = Field(default=1.0, ge=0)


class OutputRecord(BaseModel):
    spec: GenerationSpec
    prompt: str
    output_path: str
    dimensions: tuple[int, int] | None = None
    status: Literal["pending", "completed", "failed"] = "pending"
    error: str | None = None


class RunManifest(BaseModel):
    application_version: str
    creation_timestamp: datetime
    input_filename: str
    normalized_input_path: str
    normalized_input_sha256: str
    subject_hint: str | None = None
    model_id: str
    backend: Literal["flux", "mock"]
    device: str
    dtype: str
    generation_settings: GenerationSettings
    base_seed: int
    status: Literal["started", "completed", "partial", "failed"] = "started"
    outputs: list[OutputRecord]
    error: str | None = None
