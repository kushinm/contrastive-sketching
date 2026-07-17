"""End-to-end generation orchestration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PIL import Image

from . import __version__
from .artifacts import (
    create_contact_sheet,
    output_filename,
    package_run,
    safe_slug,
    save_manifest,
    write_prompts,
    write_review_csv,
)
from .backends.base import ImageEditBackend
from .image_io import normalize_image, save_normalized_input
from .prompts import build_generation_plan, build_prompt
from .schemas import GenerationSettings, OutputRecord, RunManifest

ProgressCallback = Callable[[int, int, str], None]


def run_generation(
    input_source: str | Path | Image.Image,
    subject_hint: str | None,
    base_seed: int,
    output_root: str | Path,
    backend_name: str,
    backend: ImageEditBackend,
    settings: GenerationSettings,
    progress: ProgressCallback | None = None,
    timestamp: datetime | None = None,
) -> tuple[Path, RunManifest, Path]:
    if backend_name not in {"flux", "mock"}:
        raise ValueError("backend_name must be flux or mock")
    now = timestamp or datetime.now().astimezone()
    input_filename = (
        Path(input_source).name
        if not isinstance(input_source, Image.Image)
        else "uploaded_image.png"
    )
    slug_source = subject_hint or Path(input_filename).stem
    run_dir = Path(output_root) / f"{now.strftime('%Y%m%d_%H%M%S')}_{safe_slug(slug_source)}"
    suffix = 2
    original = run_dir
    while run_dir.exists():
        run_dir = original.with_name(f"{original.name}_{suffix}")
        suffix += 1
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    normalized = normalize_image(input_source, max_side=settings.max_side)
    input_hash = save_normalized_input(normalized, run_dir / "input.png")
    specs = build_generation_plan(base_seed)
    records = [
        OutputRecord(
            spec=spec,
            prompt=build_prompt(spec, subject_hint),
            output_path=f"outputs/{output_filename(spec)}",
        )
        for spec in specs
    ]
    manifest = RunManifest(
        application_version=__version__,
        creation_timestamp=now,
        input_filename=input_filename,
        normalized_input_path="input.png",
        normalized_input_sha256=input_hash,
        subject_hint=subject_hint.strip() if subject_hint and subject_hint.strip() else None,
        model_id=settings.model_id,
        backend=backend_name,
        device="cpu" if backend_name == "mock" else settings.device,
        dtype="n/a" if backend_name == "mock" else settings.dtype,
        generation_settings=settings,
        base_seed=base_seed,
        outputs=records,
    )
    manifest_path = run_dir / "manifest.json"
    save_manifest(manifest, manifest_path)
    write_prompts(manifest, run_dir / "prompts.txt")
    write_review_csv(manifest, run_dir / "review.csv")
    completed_images: list[tuple[Image.Image, str]] = []
    for record in manifest.outputs:
        spec = record.spec
        if progress:
            progress(spec.output_index - 1, 10, f"Generating {spec.output_index}/10")
        try:
            result = backend.generate(normalized, record.prompt, spec.seed, settings)
            destination = run_dir / record.output_path
            result.convert("RGB").save(destination, format="PNG")
            record.dimensions = result.size
            record.status = "completed"
            caption = f"{spec.output_index:02d} {spec.mode.value} | {spec.variant} | {spec.strength} | seed {spec.seed}"
            completed_images.append((result.copy(), caption))
            if backend_name == "flux":
                manifest.device = getattr(backend, "resolved_device", manifest.device)
                manifest.dtype = getattr(backend, "resolved_dtype", manifest.dtype)
        except Exception as exc:  # preserve the remaining run and exact failure
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
        save_manifest(manifest, manifest_path)
    successes = sum(record.status == "completed" for record in manifest.outputs)
    if successes == 10:
        manifest.status = "completed"
        create_contact_sheet(completed_images, run_dir / "contact_sheet.png")
    elif successes:
        manifest.status = "partial"
        manifest.error = f"{10 - successes} of 10 outputs failed; see output records"
    else:
        manifest.status = "failed"
        manifest.error = "All 10 outputs failed; see output records"
    save_manifest(manifest, manifest_path)
    if progress:
        progress(10, 10, f"Run {manifest.status}")
    zip_path = package_run(run_dir)
    return run_dir, manifest, zip_path
