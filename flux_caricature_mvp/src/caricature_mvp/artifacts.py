"""Run-directory, manifest, review, contact-sheet, and ZIP artifacts."""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .schemas import GenerationSpec, RunManifest

_FILENAMES = {
    1: "01_part_emphasis_single_moderate.png",
    2: "02_part_emphasis_single_strong.png",
    3: "03_part_emphasis_relational_strong.png",
    4: "04_category_selectivity_silhouette_moderate.png",
    5: "05_category_selectivity_structure_strong.png",
    6: "06_category_selectivity_contour_strong.png",
    7: "07_iconic_simplification_moderate.png",
    8: "08_iconic_simplification_aggressive.png",
    9: "09_functional_selectivity_moderate.png",
    10: "10_functional_selectivity_strong.png",
}


def output_filename(spec: GenerationSpec) -> str:
    try:
        return _FILENAMES[spec.output_index]
    except KeyError as exc:
        raise ValueError(f"Unsupported output index: {spec.output_index}") from exc


def safe_slug(value: str, fallback: str = "subject") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:48] or fallback


def save_manifest(manifest: RunManifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_prompts(manifest: RunManifest, path: str | Path) -> None:
    sections = []
    for record in manifest.outputs:
        spec = record.spec
        sections.append(
            f"OUTPUT {spec.output_index:02d}\n"
            f"Mode: {spec.mode.value}\nVariant: {spec.variant}\n"
            f"Strength: {spec.strength}\nSeed: {spec.seed}\n\n{record.prompt}"
        )
    Path(path).write_text(
        "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(sections) + "\n", encoding="utf-8"
    )


def write_review_csv(manifest: RunManifest, path: str | Path) -> None:
    fields = [
        "output_index",
        "filename",
        "mode",
        "variant",
        "strength",
        "seed",
        "identifiable_1_to_5",
        "line_drawing_quality_1_to_5",
        "mode_adherence_1_to_5",
        "distinctiveness_1_to_5",
        "notes",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in manifest.outputs:
            spec = record.spec
            writer.writerow(
                {
                    "output_index": spec.output_index,
                    "filename": Path(record.output_path).name,
                    "mode": spec.mode.value,
                    "variant": spec.variant,
                    "strength": spec.strength,
                    "seed": spec.seed,
                    "identifiable_1_to_5": "",
                    "line_drawing_quality_1_to_5": "",
                    "mode_adherence_1_to_5": "",
                    "distinctiveness_1_to_5": "",
                    "notes": "",
                }
            )


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    contained = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return canvas


def create_contact_sheet(
    entries: list[tuple[Image.Image, str]],
    destination: str | Path,
    cell_size: tuple[int, int] = (320, 320),
) -> Path:
    if len(entries) != 10:
        raise ValueError("Contact sheet requires exactly 10 images")
    columns, rows, caption_height = 5, 2, 74
    sheet = Image.new(
        "RGB", (columns * cell_size[0], rows * (cell_size[1] + caption_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for index, (image, caption) in enumerate(entries):
        column, row = index % columns, index // columns
        x, y = column * cell_size[0], row * (cell_size[1] + caption_height)
        sheet.paste(_fit_image(image, cell_size), (x, y))
        wrapped = "\n".join(caption[pos : pos + 38] for pos in range(0, len(caption), 38))
        draw.multiline_text(
            (x + 8, y + cell_size[1] + 6), wrapped, fill="black", font=font, spacing=3
        )
        draw.rectangle(
            (x, y, x + cell_size[0] - 1, y + cell_size[1] + caption_height - 1), outline="#bbbbbb"
        )
    destination = Path(destination)
    sheet.save(destination, format="PNG")
    return destination


def package_run(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    archive_base = run_dir.parent / run_dir.name
    archive = shutil.make_archive(
        str(archive_base), "zip", root_dir=run_dir.parent, base_dir=run_dir.name
    )
    return Path(archive)
