import sys
import os
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from contrastive_clipasso import sketch

# Recreating CLIPasso paper results
# Uses RN101 (original paper default), horse.png test image
TARGET      = "target_images/cat_black.jpg"
NUM_STROKES = 16
NUM_ITER    = 500
CLIP_MODEL  = "RN101"
SEED        = 0

# Fixed
MASK_OBJECT  = True
FIX_SCALE    = True
STROKE_WIDTH = 1
IMAGE_SCALE  = 224

target_name = os.path.splitext(os.path.basename(TARGET))[0]
out = f"outputs/recreate_clipasso/{target_name}_strokes{NUM_STROKES}_iter{NUM_ITER}"
os.makedirs(out, exist_ok=True)

print(f"=== Recreating CLIPasso | horse | {NUM_STROKES} strokes | {NUM_ITER} iter ===")
result = sketch.run(
    target=TARGET,
    distractor=None,
    num_strokes=NUM_STROKES,
    num_iter=NUM_ITER,
    contrastive_weight=0.0,
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

# Loss curve
fig, ax = plt.subplots(figsize=(8, 4))
if result.get("loss_train"):
    ax.plot(result["loss_train"], alpha=0.3, label="Train loss")
if result.get("loss_eval"):
    eval_x = list(range(0, NUM_ITER, max(1, NUM_ITER // max(len(result["loss_eval"]), 1))))
    eval_x = eval_x[:len(result["loss_eval"])]
    ax.plot(eval_x, result["loss_eval"], 'r-', label="Eval loss", linewidth=2)
    ax.axhline(y=result["best_loss"], color='g', linestyle='--', alpha=0.5,
               label=f'Best: {result["best_loss"]:.4f}')
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title(f"CLIPasso Recreation | horse | {NUM_STROKES} strokes")
ax.legend()
plt.tight_layout()
plt.savefig(f"{out}/loss_curve.png")
plt.close()

print(f"\nBest loss: {result['best_loss']:.4f} at iter {result['best_iter']}")
print(f"Output saved to: {out}")

