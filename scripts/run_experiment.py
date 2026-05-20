import sys
import os
sys.path.insert(0, '.')
from contrastive_clipasso import sketch

# ============================================================
# EDIT THESE — Define all parameter combos you want to run
# ============================================================
EXPERIMENTS = [
    {
        "target": "target_images/caricatures/fox_images/fennec/fennec_16.jpeg",
        "same_cat": "target_images/caricatures/fox_images/fennec/fennec_17.jpeg",
        "diff_cat": "target_images/mouse.jpg",
        "num_strokes": 16,
        "num_iter": 1000,
        "contrastive_weight": 0.5,
    },
    {
        "target": "target_images/caricatures/fox_images/fennec/fennec_16.jpeg",
        "same_cat": "target_images/caricatures/fox_images/fennec/fennec_17.jpeg",
        "diff_cat": "target_images/mouse.jpg",
        "num_strokes": 16,
        "num_iter": 2000,
        "contrastive_weight": 0.5,
    },
    {
        "target": "target_images/caricatures/fox_images/fennec/fennec_16.jpeg",
        "same_cat": "target_images/caricatures/fox_images/fennec/fennec_17.jpeg",
        "diff_cat": "target_images/mouse.jpg",
        "num_strokes": 16,
        "num_iter": 3000,
        "contrastive_weight": 0.5,
    },
    # Add more experiments here by copying the block above
]

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
# Run all experiments
# ============================================================
all_results = []

for exp in EXPERIMENTS:
    exp_label = f"strokes{exp['num_strokes']}_iter{exp['num_iter']}_lambda{exp['contrastive_weight']}"
    
    conditions = [
        {"label": "cond1_no_distractor",  "distractor": None,           "weight": 0.0},
        {"label": "cond2_same_category",  "distractor": exp["same_cat"], "weight": exp["contrastive_weight"]},
        {"label": "cond3_diff_category",  "distractor": exp["diff_cat"], "weight": exp["contrastive_weight"]},
    ]

    for cond in conditions:
        out = f"outputs/script_outputs/{exp_label}/{cond['label']}"
        print(f"\n=== {exp_label} | {cond['label']} ===")
        result = sketch.run(
            target=exp["target"],
            distractor=cond["distractor"],
            num_strokes=exp["num_strokes"],
            num_iter=exp["num_iter"],
            contrastive_weight=cond["weight"],
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
        all_results.append({
            "exp": exp_label,
            "condition": cond["label"],
            "best_loss": result["best_loss"],
            "best_iter": result["best_iter"],
        })

# Summary
print("\n\n=== FULL SUMMARY ===")
print(f"{'Experiment':<45} {'Condition':<25} {'Best Loss':>10} {'Best Iter':>10}")
print("-" * 92)
for r in all_results:
    print(f"{r['exp']:<45} {r['condition']:<25} {r['best_loss']:>10.4f} {r['best_iter']:>10}")
