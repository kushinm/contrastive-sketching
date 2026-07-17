import json
from datetime import datetime

from PIL import Image

from caricature_mvp.artifacts import create_contact_sheet, save_manifest
from caricature_mvp.prompts import build_generation_plan, build_prompt
from caricature_mvp.schemas import GenerationSettings, OutputRecord, RunManifest


def make_manifest():
    specs = build_generation_plan(10)
    return RunManifest(
        application_version="test",
        creation_timestamp=datetime(2026, 7, 17),
        input_filename="source.jpg",
        normalized_input_path="input.png",
        normalized_input_sha256="a" * 64,
        model_id="mock-model",
        backend="mock",
        device="cpu",
        dtype="n/a",
        generation_settings=GenerationSettings(),
        base_seed=10,
        outputs=[
            OutputRecord(
                spec=spec, prompt=build_prompt(spec), output_path=f"outputs/{spec.output_index}.png"
            )
            for spec in specs
        ],
    )


def test_manifest_serialization_and_incremental_update(tmp_path):
    manifest = make_manifest()
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    assert json.loads(path.read_text())["status"] == "started"
    manifest.outputs[0].status = "completed"
    manifest.outputs[0].dimensions = (64, 64)
    save_manifest(manifest, path)
    payload = json.loads(path.read_text())
    assert payload["outputs"][0]["status"] == "completed"
    assert payload["outputs"][0]["dimensions"] == [64, 64]


def test_contact_sheet_with_ten_images(tmp_path):
    entries = [
        (Image.new("RGB", (80, 50), (index * 20, 0, 0)), f"caption {index}") for index in range(10)
    ]
    destination = create_contact_sheet(entries, tmp_path / "sheet.png", cell_size=(100, 80))
    with Image.open(destination) as sheet:
        assert sheet.size == (500, 308)
