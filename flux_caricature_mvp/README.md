# FLUX Line Caricature MVP

An independent Python MVP for generating exactly ten object-agnostic black-ink line caricatures from one reference image. It uses the same fixed drawing language across four transformation modes: part emphasis (3 outputs), category selectivity (3), iconic simplification (2), and functional selectivity (2).

This project is separate from the repository's CLIPasso approach. It does not use face landmarks, segmentation, part detectors, ControlNet, CLIP ranking, or a second vision-language model. FLUX receives the normalized source image directly as its editing reference.

## Requirements

- Python 3.10 or newer (the repository's existing `clipasso` environment uses 3.10; 3.11+ is also supported)
- For the real backend: a supported PyTorch environment and enough memory for `black-forest-labs/FLUX.2-klein-4B` (the model card describes roughly 13 GB VRAM as a practical target)
- Hugging Face access/network connectivity for the first model download

The mock backend needs no model, GPU, or network. Although the standard install includes the ML dependencies so either backend is available, mock generation never imports or loads FLUX.

## Installation

Reuse the repository's existing CLIPasso Conda environment, as requested, while installing this project as a separate package:

```bash
conda activate clipasso
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This adds the MVP's Python dependencies to the environment but does not couple its code or generation pipeline to CLIPasso.

If a released Diffusers build reports that `Flux2KleinPipeline` is unavailable, install a newer compatible Diffusers build and retry:

```bash
python -m pip install --upgrade diffusers
```

Optional defaults are documented in `.env.example`. The program reads those environment variables directly; it does not automatically parse a `.env` file.

## CLI

Fast CPU-only end-to-end run:

```bash
caricature-mvp generate path/to/input.jpg \
  --subject-hint "a compact vintage desk fan" \
  --seed 1234 \
  --output-dir runs \
  --backend mock
```

Real FLUX editing on an automatically selected device:

```bash
caricature-mvp generate path/to/input.jpg \
  --subject-hint "a compact vintage desk fan" \
  --seed 1234 \
  --output-dir runs \
  --backend flux
```

CUDA with model CPU offloading:

```bash
caricature-mvp generate path/to/input.jpg \
  --backend flux \
  --device cuda \
  --cpu-offload
```

The default model is `black-forest-labs/FLUX.2-klein-4B`. Override it with `--model-id`. Inputs may be PNG, JPEG, or WebP. EXIF orientation is applied; images are converted to RGB, resized down only when necessary, and padded with white to model-compatible dimensions without cropping.

## Gradio UI

Launch with the real backend:

```bash
caricature-mvp app --backend flux --output-dir runs
```

Launch the full UI and artifact flow without loading a model:

```bash
caricature-mvp app --backend mock --output-dir runs
```

The UI includes an image upload, optional subject hint, base seed, progress, a ten-image gallery with generation metadata, a 2-by-5 contact sheet, and the complete run ZIP.

Best results come from an image with one dominant, clearly visible subject and a reasonably uncluttered background. The MVP intentionally does not perform segmentation or background removal.

## Artifacts and reproducibility

Each run writes a timestamped directory and a sibling ZIP:

```text
runs/
  20260717_143000_desk_fan/
    input.png
    manifest.json
    prompts.txt
    contact_sheet.png
    review.csv
    outputs/
      01_part_emphasis_single_moderate.png
      02_part_emphasis_single_strong.png
      03_part_emphasis_relational_strong.png
      04_category_selectivity_silhouette_moderate.png
      05_category_selectivity_structure_strong.png
      06_category_selectivity_contour_strong.png
      07_iconic_simplification_moderate.png
      08_iconic_simplification_aggressive.png
      09_functional_selectivity_moderate.png
      10_functional_selectivity_strong.png
  20260717_143000_desk_fan.zip
```

`manifest.json` is saved before generation and after every output. It records the normalized input hash, runtime/settings, exact prompt and deterministic seed for every output, dimensions, statuses, and failures. A run with an individual failure continues and finishes as `partial`, leaving successful outputs and actionable error text inspectable. `review.csv` has ten rows with empty rating fields for manual evaluation.

Seeds are `base_seed + output_index`, so a base seed of 1234 produces 1235 through 1244.

## Tests and linting

Ordinary tests use the mock backend and do not download or import FLUX:

```bash
pytest
ruff check .
```

The opt-in smoke test loads the real model and generates one image:

```bash
RUN_FLUX_SMOKE_TEST=1 pytest -m flux_smoke tests/test_flux_smoke.py
```

## Configuration

The primary CLI intentionally exposes only model ID, device, maximum input side, and CPU offloading. Less common inference defaults live in `GenerationSettings`: four inference steps and guidance scale 1.0, matching the distilled Klein usage pattern. Dtype defaults to bfloat16 on CUDA/MPS and float32 on CPU and can be overridden with `CARICATURE_DTYPE` (`float16`, `bfloat16`, or `float32`).

The FLUX pipeline is lazy and process-scoped: a backend instance loads once on its first generation and is reused for all ten outputs. Changing model/runtime settings after that requires a new backend instance. The reference image is always passed through the pipeline's `image` argument; this implementation never silently falls back to text-to-image.

## Limitations

- Quality and identity preservation depend on the source and the model's editing behavior; there is no automated recognizability scoring.
- One visually dominant subject is assumed. Multi-object composition is out of scope.
- White padding can be visible when source dimensions are not divisible by the model alignment.
- Sequential real generation can be slow and memory-intensive.
- Mock images are deterministic edge-style placeholders for testing infrastructure, not representative model results.
- A failed/partial run has no contact sheet because a valid sheet requires exactly ten outputs; its manifest, prompts, review CSV, successful images, and ZIP are still retained.

The FLUX backend follows the official Hugging Face model-card editing interface for FLUX.2 Klein, using `Flux2KleinPipeline` with the source supplied as `image`.
