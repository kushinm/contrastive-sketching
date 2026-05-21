import sys
import os
import matplotlib.pyplot as plt
import csv
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
        "num_iter": 30000,
        "contrastive_weight": 0.7,
    },
    # {
    #     "target": "target_images/caricatures/fox_images/fennec/fennec_16.jpeg",
    #     "same_cat": "target_images/caricatures/fox_images/fennec/fennec_17.jpeg",
    #     "diff_cat": "target_images/mouse.jpg",
    #     "num_strokes": 16,
    #     "num_iter": 1000,
    #     "contrastive_weight": 0.5,
    # },
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
        {"label": "cond1_no_distractor",  "distractor": None,            "weight": 0.0},
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
            "loss_eval": result.get("loss_eval", []),
            "loss_train": result.get("loss_train", []),
        })

# ============================================================
# Save CSV
# ============================================================
os.makedirs("outputs/script_outputs", exist_ok=True)
csv_path = "outputs/script_outputs/results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["exp", "condition", "best_loss", "best_iter"])
    writer.writeheader()
    for r in all_results:
        writer.writerow({"exp": r["exp"], "condition": r["condition"],
                         "best_loss": r["best_loss"], "best_iter": r["best_iter"]})
print(f"\nResults saved to {csv_path}")

# ============================================================
# Save loss curves — one plot per condition
# ============================================================
for r in all_results:
    fig, ax = plt.subplots(figsize=(8, 4))

    if r.get("loss_train"):
        ax.plot(r["loss_train"], alpha=0.3, label="Train loss")
    if r.get("loss_eval"):
        num_iter = r["best_iter"]
        eval_x = list(range(0, len(r["loss_eval"]) * max(1, num_iter // max(len(r["loss_eval"]), 1)),
                            max(1, num_iter // max(len(r["loss_eval"]), 1))))
        eval_x = eval_x[:len(r["loss_eval"])]
        ax.plot(eval_x, r["loss_eval"], 'r-', label="Eval loss", linewidth=2)
        ax.axhline(y=r["best_loss"], color='g', linestyle='--', alpha=0.5,
                   label=f'Best: {r["best_loss"]:.4f}')

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f'{r["exp"]} | {r["condition"]}')
    ax.legend()
    plt.tight_layout()

    plot_path = f"outputs/script_outputs/{r['exp']}/{r['condition']}/loss_curve.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss curve saved to {plot_path}")

# ============================================================
# Print summary
# ============================================================
print("\n\n=== FULL SUMMARY ===")
print(f"{'Experiment':<45} {'Condition':<25} {'Best Loss':>10} {'Best Iter':>10}")
print("-" * 92)
for r in all_results:
    print(f"{r['exp']:<45} {r['condition']:<25} {r['best_loss']:>10.4f} {r['best_iter']:>10}")
