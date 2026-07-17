#!/usr/bin/env bash
# Wrapper for scripts/species_experiment.py.
#
# Sets the clipasso conda env + GPU, then hands through any extra flags.
#
# Quick usage:
#   ./scripts/run_species_experiment.sh                       # defaults (9-run smoke test)
#   ./scripts/run_species_experiment.sh --dry_run             # preview planned pairings
#   GPU=2 NUM_ITER=500 IMAGES=5 ./scripts/run_species_experiment.sh
#   ./scripts/run_species_experiment.sh --conditions within,between --num_iter 1000
#
# Env-var knobs (all optional; CLI flags still win):
#   PYTHON_BIN  path to the python to use     (default: clipasso conda env)
#   GPU         CUDA_VISIBLE_DEVICES value    (default: 2)
#   CATEGORY    --target_category             (default: fox_images)
#   CONDITIONS  --conditions                  (default: none,within,between)
#   IMAGES      --images_per_species          (default: 1)
#   NUM_ITER    --num_iter                    (default: 50)
#   NUM_PATHS   --num_paths                   (default: 16)
#   SEED        --seed                        (default: 0)
#   OUT         --output_root                 (default: outputs/species_experiment)
#   CLIP_MODEL  --clip_model_name             (default: RN101)

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/kushinm/miniconda3/envs/clipasso/bin/python}"
GPU="${GPU:-2}"
CATEGORY="${CATEGORY:-fox_images}"
CONDITIONS="${CONDITIONS:-none,within,between}"
IMAGES="${IMAGES:-1}"
NUM_ITER="${NUM_ITER:-50}"
NUM_PATHS="${NUM_PATHS:-16}"
SEED="${SEED:-0}"
OUT="${OUT:-outputs/species_experiment}"
CLIP_MODEL="${CLIP_MODEL:-RN101}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "error: PYTHON_BIN not found or not executable: $PYTHON_BIN" >&2
    echo "hint:  set PYTHON_BIN=/path/to/python or install the clipasso conda env" >&2
    exit 1
fi

echo "repo:        $REPO_ROOT"
echo "python:      $PYTHON_BIN"
echo "gpu:         $GPU"
echo "category:    $CATEGORY"
echo "conditions:  $CONDITIONS"
echo "images/sp:   $IMAGES"
echo "num_iter:    $NUM_ITER"
echo "num_paths:   $NUM_PATHS"
echo "seed:        $SEED"
echo "output:      $OUT"
echo "clip_model:  $CLIP_MODEL"
echo "extra args:  $*"
echo

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m scripts.species_experiment \
    --target_category "$CATEGORY" \
    --conditions "$CONDITIONS" \
    --images_per_species "$IMAGES" \
    --num_iter "$NUM_ITER" \
    --num_paths "$NUM_PATHS" \
    --seed "$SEED" \
    --output_root "$OUT" \
    --clip_model_name "$CLIP_MODEL" \
    "$@"
