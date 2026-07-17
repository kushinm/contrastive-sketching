"""Batch CLIPasso sketches across distractor conditions.

Runs three conditions per target image:
  - none     : no distractor (standard CLIPasso)
  - within   : distractor is a random image from a different species folder
               within the same animal category as the target
  - between  : distractor is a random image from a species folder in a
               different animal category

Category tree expected:
    <categories_root>/
        <category_a>/<species_1>/*.{jpg,jpeg,png,webp,avif,...}
        <category_a>/<species_2>/...
        <category_b>/<species_3>/...

Outputs are written under <output_root>/<condition>/<category>/<species>/<stem>/,
plus a manifest.csv at <output_root>/ with one row per run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
CONDITIONS = ("none", "within", "between")


@dataclass
class Target:
    category: str
    species: str
    path: Path

    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass
class PlannedRun:
    condition: str
    target: Target
    distractor_category: Optional[str]
    distractor_species: Optional[str]
    distractor_path: Optional[Path]
    output_dir: Path


def list_species_dirs(category_dir: Path) -> List[Path]:
    return sorted(p for p in category_dir.iterdir() if p.is_dir())


def list_images(species_dir: Path) -> List[Path]:
    return sorted(
        p for p in species_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def gather_targets(
    categories_root: Path,
    target_category: str,
    images_per_species: int,
    rng: random.Random,
) -> List[Target]:
    cat_dir = categories_root / target_category
    if not cat_dir.is_dir():
        raise FileNotFoundError(f"Target category dir not found: {cat_dir}")
    targets: List[Target] = []
    for species_dir in list_species_dirs(cat_dir):
        imgs = list_images(species_dir)
        if not imgs:
            print(f"  [warn] no images in {species_dir}, skipping")
            continue
        k = min(images_per_species, len(imgs))
        picked = rng.sample(imgs, k)
        for p in picked:
            targets.append(Target(target_category, species_dir.name, p))
    return targets


def pick_within(
    categories_root: Path, target: Target, rng: random.Random,
) -> Optional[Tuple[str, str, Path]]:
    cat_dir = categories_root / target.category
    sibling_species = [
        d for d in list_species_dirs(cat_dir) if d.name != target.species
    ]
    if not sibling_species:
        return None
    sp_dir = rng.choice(sibling_species)
    imgs = list_images(sp_dir)
    if not imgs:
        return None
    return target.category, sp_dir.name, rng.choice(imgs)


def pick_between(
    categories_root: Path, target: Target, rng: random.Random,
) -> Optional[Tuple[str, str, Path]]:
    other_cats = [
        d for d in categories_root.iterdir()
        if d.is_dir() and d.name != target.category
    ]
    if not other_cats:
        return None
    rng.shuffle(other_cats)
    for cat in other_cats:
        species_dirs = list_species_dirs(cat)
        rng.shuffle(species_dirs)
        for sp in species_dirs:
            imgs = list_images(sp)
            if imgs:
                return cat.name, sp.name, rng.choice(imgs)
    return None


def plan_runs(
    categories_root: Path,
    targets: List[Target],
    conditions: List[str],
    output_root: Path,
    rng: random.Random,
) -> List[PlannedRun]:
    plans: List[PlannedRun] = []
    for t in targets:
        for cond in conditions:
            out_dir = output_root / cond / t.category / t.species / t.stem
            if cond == "none":
                plans.append(PlannedRun(cond, t, None, None, None, out_dir))
                continue
            picker = pick_within if cond == "within" else pick_between
            chosen = picker(categories_root, t, rng)
            if chosen is None:
                print(
                    f"  [warn] cannot resolve '{cond}' distractor for "
                    f"{t.category}/{t.species}/{t.path.name} — skipping"
                )
                continue
            d_cat, d_sp, d_path = chosen
            plans.append(PlannedRun(cond, t, d_cat, d_sp, d_path, out_dir))
    return plans


def write_pairing(plan: PlannedRun, seed: int) -> None:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "condition": plan.condition,
        "target": str(plan.target.path),
        "target_category": plan.target.category,
        "target_species": plan.target.species,
        "distractor": str(plan.distractor_path) if plan.distractor_path else None,
        "distractor_category": plan.distractor_category,
        "distractor_species": plan.distractor_species,
        "seed": seed,
    }
    with open(plan.output_dir / "pairing.json", "w") as f:
        json.dump(info, f, indent=2)


def run_one(plan: PlannedRun, args: argparse.Namespace) -> Dict:
    # Import here so --dry_run doesn't require the full CLIPasso/diffvg stack.
    from contrastive_clipasso import sketch

    return sketch.run(
        target=str(plan.target.path),
        distractor=str(plan.distractor_path) if plan.distractor_path else None,
        num_strokes=args.num_paths,
        num_iter=args.num_iter,
        output_dir=str(plan.output_dir),
        seed=args.seed,
        use_gpu=not args.cpu,
        clip_model_name=args.clip_model_name,
        verbose=args.verbose,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--categories_root", default="target_images/caricatures")
    p.add_argument("--target_category", default="fox_images")
    p.add_argument(
        "--conditions", default="none,within,between",
        help="Comma-separated subset of: none,within,between",
    )
    p.add_argument("--images_per_species", type=int, default=1)
    p.add_argument("--num_iter", type=int, default=50)
    p.add_argument("--num_paths", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_root", default="outputs/species_experiment")
    p.add_argument("--clip_model_name", default="RN101")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--quiet", dest="verbose", action="store_false")
    args = p.parse_args(argv)

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        p.error(f"unknown condition(s): {bad}. Valid: {CONDITIONS}")
    args.conditions_list = conds
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    categories_root = Path(args.categories_root).resolve()
    output_root = Path(args.output_root).resolve()

    targets = gather_targets(
        categories_root, args.target_category, args.images_per_species, rng,
    )
    if not targets:
        print("No target images found — nothing to do.")
        return 1

    plans = plan_runs(
        categories_root, targets, args.conditions_list, output_root, rng,
    )

    print(f"Planned {len(plans)} runs across {len(targets)} targets "
          f"× conditions={args.conditions_list}")
    for plan in plans:
        d = f"{plan.distractor_category}/{plan.distractor_species}/{plan.distractor_path.name}" \
            if plan.distractor_path else "(none)"
        print(f"  [{plan.condition:>7}] {plan.target.species}/{plan.target.path.name}"
              f"  vs  {d}  ->  {plan.output_dir.relative_to(output_root.parent)}")

    if args.dry_run:
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"
    fieldnames = [
        "condition", "category", "species", "target_image",
        "distractor_category", "distractor_species", "distractor_image",
        "output_dir", "best_loss", "best_iter", "num_iter", "num_paths", "seed",
    ]

    with open(manifest_path, "w", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=fieldnames)
        writer.writeheader()

        for i, plan in enumerate(plans, 1):
            print(f"\n[{i}/{len(plans)}] condition={plan.condition} "
                  f"target={plan.target.species}/{plan.target.path.name}")
            write_pairing(plan, args.seed)
            try:
                hist = run_one(plan, args)
            except Exception as e:
                print(f"  [error] run failed: {e}")
                continue

            writer.writerow({
                "condition": plan.condition,
                "category": plan.target.category,
                "species": plan.target.species,
                "target_image": plan.target.path.name,
                "distractor_category": plan.distractor_category or "",
                "distractor_species": plan.distractor_species or "",
                "distractor_image": plan.distractor_path.name if plan.distractor_path else "",
                "output_dir": str(plan.output_dir),
                "best_loss": hist.get("best_loss", ""),
                "best_iter": hist.get("best_iter", ""),
                "num_iter": args.num_iter,
                "num_paths": args.num_paths,
                "seed": args.seed,
            })
            mf.flush()

    print(f"\nDone. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
