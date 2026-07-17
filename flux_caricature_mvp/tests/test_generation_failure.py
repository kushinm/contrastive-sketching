import json
from datetime import datetime

from PIL import Image

from caricature_mvp.backends.mock import MockBackend
from caricature_mvp.generation import run_generation
from caricature_mvp.schemas import GenerationSettings


class FailOneBackend(MockBackend):
    def generate(self, input_image, prompt, seed, settings):
        if seed == 103:
            raise RuntimeError("intentional test failure")
        return super().generate(input_image, prompt, seed, settings)


def test_individual_failure_produces_useful_partial_manifest(tmp_path):
    run_dir, manifest, _ = run_generation(
        input_source=Image.new("RGB", (80, 80), "green"),
        subject_hint=None,
        base_seed=100,
        output_root=tmp_path,
        backend_name="mock",
        backend=FailOneBackend(),
        settings=GenerationSettings(max_side=128),
        timestamp=datetime(2026, 7, 17, 12, 0, 0),
    )
    assert manifest.status == "partial"
    failed = [record for record in manifest.outputs if record.status == "failed"]
    assert len(failed) == 1
    assert failed[0].spec.output_index == 3
    assert "intentional test failure" in failed[0].error
    disk_manifest = json.loads((run_dir / "manifest.json").read_text())
    assert disk_manifest["status"] == "partial"
    assert disk_manifest["outputs"][2]["status"] == "failed"
