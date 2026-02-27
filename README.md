# Contrastive CLIPasso

Generate sketches of images using CLIP-guided stroke optimization — with an optional **contrastive mode** that makes sketches maximally distinguishable from a distractor image.

Built on the ideas from [CLIPasso (Vinker et al.)](https://clipasso.github.io/clipasso/), reorganised and extended.

## Modes

| Mode | Input | Output |
|------|-------|--------|
| **Normal** | 1 image | Sketch that captures the target image |
| **Contrastive** | 2 images (target + distractor) | Sketch of the target that is visually distinct from the distractor |

The contrastive mode is like Pictionary: draw one image so a viewer would never confuse it with the other.

---

## Installation

### Compatibility at a Glance

| Component | Recommended | Notes |
|-----------|-------------|-------|
| **Python** | 3.10 or 3.11 | 3.8 dropped by PyTorch 2.5+; 3.12 works but less tested |
| **PyTorch** | 2.5.1+cu124 | cu124 builds work on CUDA 12.4–12.6 drivers (forward compat) |
| **torchvision** | 0.20.1+cu124 | Must match PyTorch version exactly |
| **numpy** | ≥1.23, <2.0 | numpy 2.0 has ABI-breaking changes that crash PyTorch & scikit-image |
| **Pillow** | ≥9.0, <11.0 | Code handles the `BICUBIC` → `Resampling.BICUBIC` deprecation |
| **diffvg** | CPU build | CUDA build broken on CUDA 12; CPU rendering is fine (see below) |
| **CLIP** | latest from git | Stable across PyTorch versions |

> **Why CPU-only diffvg?** diffvg was written for CUDA 10/11 and uses deprecated `thrust` APIs
> removed in CUDA 12. Its CUDA build will fail on modern systems. This is fine because diffvg
> only does SVG rasterisation (fast on CPU) — the actual GPU-heavy work (CLIP features,
> backprop through augmentations) runs through PyTorch's own CUDA runtime.

### Prerequisites

- Linux (Ubuntu 20.04+ recommended) or macOS
- CUDA 12.x driver + GPU (for PyTorch; `nvidia-smi` should work)
- `cmake ≥ 3.12` (for building diffvg)
- conda or venv for environment isolation

### Step-by-step

```bash
# 0. Check your system first
python3 check_env.py

# 1. Create a conda environment
conda create -n clipasso python=3.10 -y
conda activate clipasso

# 2. Install PyTorch + torchvision matching your CUDA driver
#    CUDA 12.4–12.6 → use cu124 builds (forward compatible with 12.6 driver)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

#    Alternative: CUDA 12.6 native builds (PyTorch 2.6+)
#    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

# 3. Install CLIP
pip install git+https://github.com/openai/CLIP.git

# 4. Build and install diffvg (CPU-only — recommended for CUDA 12)
bash install_diffvg.sh
#    If you want to try GPU-accelerated diffvg (may fail on CUDA 12):
#    bash install_diffvg.sh --cuda

# 5. Install remaining dependencies
pip install -r requirements.txt

# 6. Verify everything works
python check_env.py --full

# 7. (Optional) Download U2Net weights for background masking
pip install gdown
gdown "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ" -O "U2Net_/saved_models/"
```

### Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `No module named 'pydiffvg'` | diffvg not built or not on path | Re-run `bash install_diffvg.sh`. If it built but import fails, add the build dir: `export PYTHONPATH="diffvg/build/lib.linux-x86_64-3.10:$PYTHONPATH"` |
| diffvg CUDA build fails | CUDA 12 removed APIs that diffvg uses | Use CPU build: `bash install_diffvg.sh` (no `--cuda`) |
| `ImportError: numpy.core.multiarray` | numpy 2.0 ABI break | `pip install 'numpy>=1.23,<2.0'` |
| `PIL.Image has no attribute BICUBIC` | Pillow 10+ deprecation | Already handled by compat shim. If you see this, update to latest code. |
| PyTorch doesn't see GPU | Wrong CUDA build or driver mismatch | Check `python -c "import torch; print(torch.cuda.is_available())"`. Use cu124 for CUDA 12.x drivers. |
| CLIP import error | Not installed from git | `pip install git+https://github.com/openai/CLIP.git` |
| CUDA OOM | Too many augmentations or high resolution | Reduce `--num_aug_clip 2` or `--image_scale 128` |
| `Python 3.8 + torch cu124` doesn't exist | PyTorch 2.5 dropped Python 3.8 | Upgrade to Python 3.10+ |

### PyTorch + CUDA Version Matrix

| Your CUDA Driver | PyTorch Build | Install Command |
|-------------------|---------------|-----------------|
| 12.4–12.6 | cu124 | `pip install torch==2.5.1 --index-url .../cu124` |
| 12.6+ | cu126 | `pip install torch==2.6.0 --index-url .../cu126` |
| 11.8 | cu118 | `pip install torch==2.5.1 --index-url .../cu118` |
| No GPU | cpu | `pip install torch==2.5.1 --index-url .../cpu` |

---

## Quick Start

### Python API

```python
from contrastive_clipasso import sketch

# Normal mode — single image
result = sketch.run(
    target="target_images/horse.jpg",
    num_strokes=16,
    num_iter=1000,
    output_dir="outputs/horse",
)

# Contrastive mode — target + distractor
result = sketch.run(
    target="target_images/horse.jpg",
    distractor="target_images/dog.jpg",
    num_strokes=16,
    num_iter=1000,
    contrastive_weight=0.5,
    output_dir="outputs/horse_vs_dog",
)
```

### CLI

```bash
# Normal mode
python -m contrastive_clipasso.sketch \
    target_images/horse.jpg \
    --num_strokes 16 --num_iter 1000 \
    --output_dir outputs/horse

# Contrastive mode
python -m contrastive_clipasso.sketch \
    target_images/horse.jpg \
    --distractor target_images/dog.jpg \
    --num_strokes 16 --num_iter 1000 \
    --contrastive_weight 0.5 \
    --output_dir outputs/horse_vs_dog
```

### Notebook

See `notebooks/contrastive_sketch.ipynb` for an interactive walkthrough.

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_strokes` | 16 | Number of Bézier strokes (fewer = more abstract) |
| `num_iter` | 1000 | Optimisation iterations |
| `contrastive_weight` | 0.5 | How much to repel from distractor (0 = normal mode) |
| `image_scale` | 224 | Resolution for optimisation |
| `width` | 1.5 | Stroke width |
| `clip_model_name` | RN101 | CLIP backbone for conv loss |
| `clip_conv_layer_weights` | 0,0,1.0,1.0,0 | Per-layer conv loss weights |
| `clip_fc_loss_weight` | 0.1 | FC-layer loss weight |
| `mask_object` | False | Use U2Net to remove background |
| `fix_scale` | False | Pad non-square images to square |
| `seed` | 0 | Random seed |

---

## How Contrastive Mode Works

Normal CLIPasso minimises the CLIP feature distance between the sketch and the target:

```
L_normal = dist(CLIP(sketch), CLIP(target))
```

Contrastive mode adds a repulsion term that pushes the sketch away from the distractor:

```
L_contrastive = dist(CLIP(sketch), CLIP(target)) − λ · dist(CLIP(sketch), CLIP(distractor))
```

Minimising this jointly:
1. **Attracts** the sketch toward the target (so it still looks like the target)
2. **Repels** the sketch from the distractor (so it emphasises features unique to the target)

The `contrastive_weight` (λ) controls the trade-off. Higher values produce more exaggerated
differences but may distort the sketch. Values of 0.3–0.7 work well in practice.

---

## Project Structure

```
contrastive-clipasso/
├── check_env.py                      # Pre/post-install diagnostics
├── install_diffvg.sh                 # diffvg builder with CUDA 12 handling
├── requirements.txt                  # Pinned deps (install PyTorch separately)
├── setup.py
├── contrastive_clipasso/
│   ├── __init__.py
│   ├── __main__.py                   # CLI entry
│   ├── compat.py                     # Version shims (Pillow, numpy, diffvg)
│   ├── config.py                     # Dataclass-based config
│   ├── losses.py                     # ContrastiveCLIPLoss (the core addition)
│   ├── painter.py                    # Stroke renderer (diffvg wrapper)
│   ├── sketch.py                     # Training loop + run() API
│   └── utils.py                      # Image I/O, masking, visualisation
├── notebooks/
│   └── contrastive_sketch.ipynb      # Interactive demo
├── target_images/                    # Your input images
└── U2Net_/saved_models/              # Background masking weights (optional)
```

---

## License

Research use. Built on CLIPasso (MIT) and diffvg (Apache 2.0).
