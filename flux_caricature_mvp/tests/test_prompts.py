import pytest

from caricature_mvp.prompts import build_generation_plan, build_prompt
from caricature_mvp.schemas import CaricatureMode


@pytest.mark.parametrize("spec", build_generation_plan(7))
def test_every_prompt_has_fixed_line_drawing_constraints(spec):
    prompt = build_prompt(spec, "a red vintage object")
    for phrase in (
        "simple black-ink line caricature",
        "same specific subject",
        "pure white background",
        "Do not use color",
        "Do not distort every feature",
    ):
        assert phrase in prompt
    assert "metaphor" not in prompt.lower()
    assert "a red vintage object" in prompt


def test_mode_prompts_contain_intended_transformation_language():
    prompts = {mode: [] for mode in CaricatureMode}
    for spec in build_generation_plan(0):
        prompts[spec.mode].append(build_prompt(spec))
    assert all("component" in prompt for prompt in prompts[CaricatureMode.PART_EMPHASIS])
    assert all("category" in prompt for prompt in prompts[CaricatureMode.CATEGORY_SELECTIVITY])
    assert all(
        "minimum" in prompt or "simplify" in prompt
        for prompt in prompts[CaricatureMode.ICONIC_SIMPLIFICATION]
    )
    assert all(
        "function" in prompt or "afford" in prompt
        for prompt in prompts[CaricatureMode.FUNCTIONAL_SELECTIVITY]
    )
