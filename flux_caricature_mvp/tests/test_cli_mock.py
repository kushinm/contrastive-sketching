import csv
import json
import zipfile

from PIL import Image
from typer.testing import CliRunner

from caricature_mvp.cli import cli


def test_mock_cli_creates_complete_artifact_directory(tmp_path):
    input_path = tmp_path / "input.jpg"
    Image.new("RGB", (120, 80), "orange").save(input_path)
    output_root = tmp_path / "runs"
    result = CliRunner().invoke(
        cli,
        [
            "generate",
            str(input_path),
            "--backend",
            "mock",
            "--seed",
            "50",
            "--output-dir",
            str(output_root),
        ],
    )
    assert result.exit_code == 0, result.output
    run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert {
        "input.png",
        "manifest.json",
        "prompts.txt",
        "contact_sheet.png",
        "review.csv",
        "outputs",
    } <= {path.name for path in run_dir.iterdir()}
    assert len(list((run_dir / "outputs").glob("*.png"))) == 10
    payload = json.loads((run_dir / "manifest.json").read_text())
    assert payload["status"] == "completed"
    assert len(payload["outputs"]) == 10
    with (run_dir / "review.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 10
    archive = run_dir.with_suffix(".zip")
    assert archive.exists()
    with zipfile.ZipFile(archive) as zipped:
        assert f"{run_dir.name}/manifest.json" in zipped.namelist()
