"""Deterministic plan and centralized prompt construction."""

from __future__ import annotations

from .schemas import CaricatureMode, GenerationSpec

BASE_PROMPT = """Transform the main subject in the reference image into a simple black-ink line caricature. It must remain immediately identifiable as the same specific subject, not merely another example from the same category. Preserve its original viewpoint, orientation, and important instance-level traits unless the caricature instruction explicitly changes a proportion.

Draw it using sparse, confident black contour lines and only a few necessary internal lines on a pure white background. Use selective exaggeration. Do not distort every feature.

Do not use color, gray wash, gradients, realistic shading, cross-hatching, detailed texture, large filled black regions, text, labels, borders, scenery, or additional objects."""

_PLAN: tuple[tuple[CaricatureMode, str, str, str], ...] = (
    (
        CaricatureMode.PART_EMPHASIS,
        "single",
        "moderate",
        "Select the single most distinctive visible component of this specific subject and enlarge or accentuate it moderately. Keep the other parts restrained and preserve their instance-level traits.",
    ),
    (
        CaricatureMode.PART_EMPHASIS,
        "single",
        "strong",
        "Select the single most distinctive visible component of this specific subject and exaggerate it strongly but coherently. Do not enlarge or distort every other component.",
    ),
    (
        CaricatureMode.PART_EMPHASIS,
        "relational",
        "strong",
        "Identify two distinctive visible components and strongly emphasize their relationship or contrast in size, length, curvature, or placement while preserving the same subject.",
    ),
    (
        CaricatureMode.CATEGORY_SELECTIVITY,
        "silhouette",
        "moderate",
        "Moderately exaggerate the overall silhouette and proportions that immediately signal the subject's broad category, while retaining visible traits unique to this instance.",
    ),
    (
        CaricatureMode.CATEGORY_SELECTIVITY,
        "structure",
        "strong",
        "Strongly exaggerate the arrangement and relative scale of category-defining components. Preserve this instance rather than substituting a generic category example.",
    ),
    (
        CaricatureMode.CATEGORY_SELECTIVITY,
        "contour",
        "strong",
        "Strongly clarify category-defining contour and proportion cues across the whole subject without simply enlarging one isolated part.",
    ),
    (
        CaricatureMode.ICONIC_SIMPLIFICATION,
        "contour",
        "moderate",
        "Moderately simplify the subject: retain its major outer contour and only a few high-information internal lines, omitting generic texture and low-information detail.",
    ),
    (
        CaricatureMode.ICONIC_SIMPLIFICATION,
        "minimal",
        "aggressive",
        "Aggressively reduce the subject to the minimum collection of confident contours and internal marks needed to recognize this specific instance.",
    ),
    (
        CaricatureMode.FUNCTIONAL_SELECTIVITY,
        "functional",
        "moderate",
        "Moderately emphasize components and relationships that communicate what the subject does, how it behaves, or what it affords. For a natural subject, emphasize characteristic behavior or structural function.",
    ),
    (
        CaricatureMode.FUNCTIONAL_SELECTIVITY,
        "functional",
        "strong",
        "Strongly exaggerate the relationship between components that communicates function, affordance, characteristic behavior, or structural function, while keeping the subject identifiable.",
    ),
)


def build_generation_plan(base_seed: int) -> list[GenerationSpec]:
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    return [
        GenerationSpec(
            output_index=index,
            mode=mode,
            variant=variant,
            strength=strength,
            instruction=instruction,
            seed=base_seed + index,
        )
        for index, (mode, variant, strength, instruction) in enumerate(_PLAN, start=1)
    ]


def build_prompt(spec: GenerationSpec, subject_hint: str | None = None) -> str:
    hint = (
        subject_hint.strip()
        if subject_hint and subject_hint.strip()
        else "None provided. Infer the single dominant subject from the reference image."
    )
    return (
        f"{BASE_PROMPT}\n\nOptional subject hint: {hint}\n\n"
        f"Caricature mode: {spec.mode.value.replace('_', ' ')}. "
        f"Exaggeration strength: {spec.strength}.\n"
        f"Caricature instruction: {spec.instruction}"
    )
