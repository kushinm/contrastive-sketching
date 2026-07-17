"""Minimal Gradio Blocks UI."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .artifacts import output_filename
from .backends import FluxKleinBackend, MockBackend
from .generation import run_generation
from .schemas import GenerationSettings


def create_app(output_dir: str | Path, backend_name: str, settings: GenerationSettings):
    import gradio as gr

    if backend_name not in {"flux", "mock"}:
        raise ValueError("backend_name must be flux or mock")
    backend = MockBackend() if backend_name == "mock" else FluxKleinBackend()

    def generate(
        image: Image.Image | None,
        subject_hint: str,
        seed: int,
        progress=gr.Progress(),  # noqa: B008 - Gradio injects progress via this sentinel.
    ):
        if image is None:
            raise gr.Error("Upload a PNG, JPEG, or WebP image first.")

        def update(done: int, total: int, message: str) -> None:
            progress(done / total, desc=message)

        run_dir, manifest, zip_path = run_generation(
            input_source=image,
            subject_hint=subject_hint,
            base_seed=int(seed),
            output_root=output_dir,
            backend_name=backend_name,
            backend=backend,
            settings=settings,
            progress=update,
        )
        if manifest.status != "completed":
            raise gr.Error(f"Run ended {manifest.status}. Inspect {run_dir / 'manifest.json'}")
        gallery = []
        for record in manifest.outputs:
            spec = record.spec
            caption = f"{spec.output_index:02d} · {spec.mode.value} · {spec.variant} · {spec.strength} · seed {spec.seed}"
            gallery.append((str(run_dir / "outputs" / output_filename(spec)), caption))
        return gallery, str(run_dir / "contact_sheet.png"), str(zip_path), f"Completed: {run_dir}"

    with gr.Blocks(title="FLUX Line Caricature MVP") as demo:
        gr.Markdown(
            "# Object-agnostic line caricatures\n"
            "Upload one dominant, clearly visible subject. A reasonably uncluttered background works best. "
            "The app does not segment or remove backgrounds."
        )
        with gr.Row():
            image_input = gr.Image(type="pil", label="Reference image")
            with gr.Column():
                hint_input = gr.Textbox(
                    label="Optional subject hint", placeholder="a compact vintage desk fan"
                )
                seed_input = gr.Number(value=1234, precision=0, minimum=0, label="Base seed")
                generate_button = gr.Button("Generate 10 caricatures", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
        gallery = gr.Gallery(label="10 interpretations", columns=5, rows=2, object_fit="contain")
        contact_sheet = gr.Image(label="2 × 5 contact sheet", type="filepath")
        archive = gr.File(label="Download complete run ZIP")
        generate_button.click(
            generate,
            inputs=[image_input, hint_input, seed_input],
            outputs=[gallery, contact_sheet, archive, status],
        )
    return demo
