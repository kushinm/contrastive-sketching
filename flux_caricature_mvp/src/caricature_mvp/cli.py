"""Typer command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .backends import FluxKleinBackend, MockBackend
from .config import settings_from_environment
from .generation import run_generation

cli = typer.Typer(no_args_is_help=True, help="Generate object-agnostic FLUX line caricatures.")


@cli.command()
def generate(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    subject_hint: Annotated[
        str | None, typer.Option(help="Optional description of the dominant subject.")
    ] = None,
    seed: Annotated[int, typer.Option(min=0, help="Base seed; outputs use seed + index.")] = 1234,
    output_dir: Annotated[Path, typer.Option(help="Root directory for timestamped runs.")] = Path(
        "runs"
    ),
    backend: Annotated[str, typer.Option(help="Inference backend: flux or mock.")] = "flux",
    model_id: Annotated[str | None, typer.Option(help="Hugging Face model ID override.")] = None,
    device: Annotated[str | None, typer.Option(help="auto, cuda, cpu, or mps.")] = None,
    max_side: Annotated[
        int | None, typer.Option(min=64, help="Maximum normalized image side.")
    ] = None,
    cpu_offload: Annotated[bool | None, typer.Option("--cpu-offload/--no-cpu-offload")] = None,
) -> None:
    """Generate exactly 10 caricature interpretations for INPUT_PATH."""
    backend = backend.lower()
    if backend not in {"flux", "mock"}:
        raise typer.BadParameter("must be 'flux' or 'mock'", param_hint="--backend")
    settings = settings_from_environment(
        model_id=model_id, device=device, max_side=max_side, cpu_offload=cpu_offload
    )
    implementation = MockBackend() if backend == "mock" else FluxKleinBackend()

    def report(done: int, total: int, message: str) -> None:
        typer.echo(f"[{done}/{total}] {message}")

    run_dir, manifest, zip_path = run_generation(
        input_source=input_path,
        subject_hint=subject_hint,
        base_seed=seed,
        output_root=output_dir,
        backend_name=backend,
        backend=implementation,
        settings=settings,
        progress=report,
    )
    typer.echo(f"Run directory: {run_dir}")
    typer.echo(f"ZIP archive: {zip_path}")
    if manifest.status != "completed":
        typer.echo(f"Run status: {manifest.status}", err=True)
        raise typer.Exit(code=1)


@cli.command()
def app(
    output_dir: Annotated[Path, typer.Option(help="Root directory for generated runs.")] = Path(
        "runs"
    ),
    backend: Annotated[str, typer.Option(help="Default backend: flux or mock.")] = "flux",
    model_id: Annotated[str | None, typer.Option()] = None,
    device: Annotated[str | None, typer.Option()] = None,
    max_side: Annotated[int | None, typer.Option(min=64)] = None,
    cpu_offload: Annotated[bool | None, typer.Option("--cpu-offload/--no-cpu-offload")] = None,
    share: Annotated[bool, typer.Option(help="Create a Gradio share link.")] = False,
) -> None:
    """Launch the minimal Gradio interface."""
    from .app import create_app

    settings = settings_from_environment(
        model_id=model_id, device=device, max_side=max_side, cpu_offload=cpu_offload
    )
    create_app(output_dir=output_dir, backend_name=backend, settings=settings).launch(share=share)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
