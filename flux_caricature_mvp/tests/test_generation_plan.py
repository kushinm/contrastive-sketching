from collections import Counter

from caricature_mvp.artifacts import output_filename
from caricature_mvp.prompts import build_generation_plan
from caricature_mvp.schemas import CaricatureMode


def test_plan_has_required_count_distribution_and_seeds():
    plan = build_generation_plan(100)
    assert len(plan) == 10
    assert Counter(spec.mode for spec in plan) == {
        CaricatureMode.PART_EMPHASIS: 3,
        CaricatureMode.CATEGORY_SELECTIVITY: 3,
        CaricatureMode.ICONIC_SIMPLIFICATION: 2,
        CaricatureMode.FUNCTIONAL_SELECTIVITY: 2,
    }
    assert [spec.seed for spec in plan] == list(range(101, 111))
    assert plan == build_generation_plan(100)


def test_output_filenames_are_stable_and_unique():
    names = [output_filename(spec) for spec in build_generation_plan(0)]
    assert len(names) == len(set(names)) == 10
    assert names[0] == "01_part_emphasis_single_moderate.png"
    assert names[-1] == "10_functional_selectivity_strong.png"
