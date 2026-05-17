import sys
import os
sys.path.insert(0, '.')
from contrastive_clipasso import sketch

# ============================================================
# EDIT THESE — Images
# ============================================================
TARGET_IMAGE       = "target_images/caricatures/fox_images/fennec/fennec_16.jpeg"
SAME_CAT_DISTRACTOR = "target_images/caricatures/fox_images/fennec/fennec_17.jpeg"
DIFF_CAT_DISTRACTOR = "target_images/mouse.jpg"

# ============================================================
# EDIT THESE — Parameters
# ============================================================
NUM_STROKES        = 16
NUM_ITER           = 500
CONTRASTIVE_WEIGHT = 0.5

# ============================================================
# Fixed — don't change these
# ============================================================
MASK_OBJECT  = True
FIX_SCALE    = True
STROKE_WIDTH = 1
IMAGE_SCALE  = 224
SEED         = 42
CLIP_MODEL   = "ViT-B/32"

# ============================================================
# Run 3 conditions
# ============================================================
conditions = [
    {
        "label": "cond1_no_distractor",
        "distractor": None,
        "contrastive_weight": 0.0,
    },
    {
        "label": "cond2_same_category",
        "distractor": SAME_CAT_DISTRACTOR,
        "contrastive_weight": CONTRASTIVE_WEIGHT,
    },
    {
        "label": "cond3_diff_category",
        "distractor": DIFF_CAT_DISTRACTOR,
        "contrastive_weight": CONTRASTIVE_WEIGHT,
    },
]

results = {}

for cond in conditions:
    print(f"\n=== {cond['label']} ===")
    out = f"outputs/script_outputs/{cond['label']}"
    results[cond["label"]] = sketch.run(
        target=TARGET_IMAGE,
        distractor=cond["distractor"],
        num_strokes=NUM_STROKES,
        num_iter=NUM_ITER,
        contrastive_weight=cond["contrastive_weight"],
        output_dir=out,
        image_scale=IMAGE_SCALE,
        width=STROKE_WIDTH,
        seed=SEED,
        clip_model_name=CLIP_MODEL,
        mask_object=MASK_OBJECT,
        fix_scale=FIX_SCALE,
        use_gpu=True,
        verbose=True,
    )

# Summary
print("\n\n=== SUMMARY ===")
print(f"{'Condition':<30} {'Mode':<20} {'Best Loss':>10} {'Best Iter':>10}")
print("-" * 72)
for label, res in results.items():
    mode = "Normal" if "no_distractor" in label else "Contrastive"
    print(f"{label:<30} {mode:<20} {res['best_loss']:>10.4f} {res['best_iter']:>10}")

