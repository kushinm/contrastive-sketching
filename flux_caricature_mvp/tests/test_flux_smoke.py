import os

import pytest
from PIL import Image

from caricature_mvp.backends.flux_klein import FluxKleinBackend
from caricature_mvp.prompts import build_generation_plan, build_prompt
from caricature_mvp.schemas import GenerationSettings


@pytest.mark.flux_smoke
@pytest.mark.skipif(os.getenv("RUN_FLUX_SMOKE_TEST") != "1", reason="set RUN_FLUX_SMOKE_TEST=1")
def test_real_flux_single_image_smoke():
    settings = GenerationSettings(max_side=256)
    source = Image.new("RGB", (256, 256), "white")
    spec = build_generation_plan(123)[0]
    result = FluxKleinBackend().generate(
        source, build_prompt(spec, "a simple red ball"), spec.seed, settings
    )
    assert isinstance(result, Image.Image)
    assert result.width > 0 and result.height > 0
