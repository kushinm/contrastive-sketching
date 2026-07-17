# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Contrastive CLIPasso: CLIP-guided Bézier-stroke sketch generation built on the ideas of [CLIPasso](https://clipasso.github.io/clipasso/), reorganised into a small package and extended with a **contrastive mode** (target + distractor → sketch that repels the distractor). The main package is [contrastive_clipasso/](contrastive_clipasso/); [diffvg/](diffvg/) is the vendored differentiable SVG renderer and [U2Net_/](U2Net_/) is the vendored background-masking model (weights not committed).

## Environment

- Python 3.10/3.11, PyTorch 2.5.1 with cu124 (CUDA 12.4–12.6 drivers), numpy `>=1.23,<2.0` (numpy 2.0 ABI-breaks PyTorch and scikit-image), Pillow `>=9.0,<11.0`.
- `diffvg` is built **CPU-only by default** — its CUDA path uses APIs removed in CUDA 12 and will fail to compile. CPU rasterisation is fast and GPU is still used for CLIP via PyTorch. The `Painter._render` loop ([contrastive_clipasso/painter.py:106](contrastive_clipasso/painter.py#L106)) catches `RuntimeError: not compiled with GPU` at runtime and transparently re-serialises shapes onto CPU, so code elsewhere should not assume a particular diffvg backend.
- `contrastive_clipasso/__init__.py` imports [compat.py](contrastive_clipasso/compat.py) first — it patches deprecated `numpy` aliases and resolves `PIL.Image.BICUBIC` vs `Image.Resampling.BICUBIC`. Import compat before anything that touches numpy/PIL.

## Common commands

```bash
# Diagnose environment before/after install
python3 check_env.py                          # pre-install checks
python3 check_env.py --full                   # full post-install verification

# Build / install diffvg
bash install_diffvg.sh                        # CPU-only (recommended, works on CUDA 12)
bash install_diffvg.sh --cuda                 # GPU build (will likely fail on CUDA 12)

# Python deps (install PyTorch FIRST from the cu124 index)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt

# Run a sketch (CLI)
python -m contrastive_clipasso.sketch target_images/this_cat.jpg \
    --num_strokes 16 --num_iter 1000 --output_dir outputs/cat
# Contrastive mode
python -m contrastive_clipasso.sketch target_images/this_cat.jpg \
    --distractor target_images/this_dog.jpeg \
    --contrastive_weight 0.5 --output_dir outputs/cat_vs_dog

# Install U2Net weights (only needed with --mask_object)
gdown "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ" -O U2Net_/saved_models/
```

There is no test suite, linter, or CI configured.

## Architecture

The sketching pipeline is 4 files in [contrastive_clipasso/](contrastive_clipasso/), loosely coupled through a single [`SketchConfig`](contrastive_clipasso/config.py) dataclass:

1. **[config.py](contrastive_clipasso/config.py)** — one dataclass with all parameters, plus `from_cli()` for argparse and a `__post_init__` that resolves `device`, seeds RNGs, and auto-enables contrastive mode when a distractor is passed without an explicit weight. `is_contrastive` is the canonical check.
2. **[sketch.py](contrastive_clipasso/sketch.py)** — `train(cfg)` is the optimisation loop; `run(...)` is the Python API wrapper; `main()` is the CLI entry. Loss history is returned as a dict and also saved to `config.npy`. Early-stopping uses a two-strike rule on `|eval_loss − best| ≤ min_delta`.
3. **[losses.py](contrastive_clipasso/losses.py)** — `ContrastiveCLIPLoss` extracts per-stage conv features and an FC feature from a frozen CLIP visual encoder (ResNet via explicit stem/layer1-4 taps, ViT via forward hooks on transformer blocks), computes an **attraction** loss against the target and, if a distractor is supplied, subtracts a weighted **repulsion** loss against it. `clip_conv_layer_weights` is auto-padded/truncated to match the backbone's stage count. `forward` returns a dict of named components that the training loop sums — don't assume a scalar.
4. **[painter.py](contrastive_clipasso/painter.py)** — `Painter` is an `nn.Module` holding the optimisable Bézier `points` tensors and `ShapeGroup`s. Two non-obvious invariants:
   - `self.device` (where CLIP runs) and `self.render_device` (where diffvg renders) can differ. `get_image()` always renders on `render_device` and then `.to(self.device)` before returning. If you add new shape tensors, put them on `render_device`.
   - Stroke seed points are picked by saliency — CLIP gradCAM (RN backbones) or CLIP attention rollout (ViT backbones), optionally intersected with an XDoG edge map. `_patch_clip_attn_probs` monkey-patches each ViT block's `attention()` to stash attention weights, replacing behaviour the original CLIPasso got from a forked CLIP.
   - `PainterOptimizer` wraps separate Adam optimisers for points and (optionally, when `force_sparse > 0`) colors.

[utils.py](contrastive_clipasso/utils.py) handles image I/O, optional U2Net masking, SVG re-rendering for final outputs, and `plot_batch` snapshots. `get_mask_u2net` searches multiple candidate paths for `U2Net_/` so the pipeline works regardless of CWD.

## Outputs layout

Each run writes into `cfg.output_dir`:
- `input.png`, `distractor.png` (contrastive only)
- `jpg_logs/iter{N}.jpg` and `svg_logs/svg_iter{N}.svg` every `save_interval`
- `best_iter.{jpg,svg}` updated whenever eval loss improves
- `final_sketch.png`, `final_svg.svg`, `config.npy` (dict of config + loss history)
