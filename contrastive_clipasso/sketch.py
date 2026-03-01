"""
Main entry point for Contrastive CLIPasso.

Provides:
  - run()           High-level API for Python usage
  - train()         Core training loop
  - CLI via __main__
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
from tqdm.auto import trange

from .config import SketchConfig
from .losses import ContrastiveCLIPLoss
from .painter import Painter, PainterOptimizer
from . import utils
from .compat import check_diffvg


def _setup_diffvg(cfg: SketchConfig):
    """Configure pydiffvg with proper GPU/CPU fallback."""
    available, gpu_ok = check_diffvg()
    if not available:
        raise ImportError(
            "pydiffvg is not installed. Run: bash install_diffvg.sh\n"
            "See README.md for full installation instructions."
        )
    
    import pydiffvg
    use_gpu = cfg.use_gpu and gpu_ok and torch.cuda.is_available()
    pydiffvg.set_use_gpu(use_gpu)
    
    # When diffvg renders on CPU, its output tensors must be on CPU too.
    # pydiffvg.set_device controls where render buffers are allocated —
    # setting it to a CUDA device with use_gpu=False causes device mismatches.
    # cfg.device stays as CUDA for everything else (CLIP, backprop, etc).
    if use_gpu:
        pydiffvg.set_device(cfg.device)
    else:
        pydiffvg.set_device(torch.device("cpu"))
    
    if cfg.use_gpu and not gpu_ok:
        print("  Note: diffvg using CPU rendering (GPU not available in diffvg build).")
        print("  This is normal for CUDA 12 installs — CLIP still uses GPU via PyTorch.")


def train(cfg: SketchConfig, verbose: bool = True) -> Dict:
    """
    Run the sketch optimisation loop.

    Returns a dict of config + loss history for later analysis.
    """
    # Setup diffvg
    _setup_diffvg(cfg)

    if verbose:
        mode_str = "CONTRASTIVE" if cfg.is_contrastive else "NORMAL"
        print(f"\n{'='*60}")
        print(f"  Contrastive CLIPasso — {mode_str} mode")
        print(f"  Target:     {cfg.target}")
        if cfg.is_contrastive:
            print(f"  Distractor: {cfg.distractor}")
            print(f"  λ (weight): {cfg.contrastive_weight}")
        print(f"  Strokes:    {cfg.num_paths}")
        print(f"  Iterations: {cfg.num_iter}")
        print(f"  Device:     {cfg.device}")
        print(f"  Output:     {cfg.output_dir}")
        print(f"{'='*60}\n")

    # Load images
    target_im, mask = utils.load_target(cfg)
    utils.save_input_image(target_im, cfg.output_dir)

    distractor_im = None
    if cfg.is_contrastive:
        distractor_im = utils.load_distractor(cfg)
        # Save distractor for reference
        d_np = distractor_im[0].cpu().permute(1, 2, 0).numpy()
        d_np = (d_np * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(d_np).save(os.path.join(cfg.output_dir, "distractor.png"))

    # Build loss, renderer, optimiser
    loss_fn = ContrastiveCLIPLoss(cfg)
    renderer = Painter(cfg, target_im, mask).to(cfg.device)
    optimizer = PainterOptimizer(cfg, renderer)

    # Initialise strokes
    renderer.set_random_noise(0)
    renderer.init_image(stage=0)
    optimizer.init_optimizers()

    # Tracking
    history = {"loss_eval": [], "loss_train": []}
    best_loss = float("inf")
    best_iter = 0
    min_delta = 1e-5
    terminate = False

    # ── Main loop ─────────────────────────────────────────────────

    pbar = trange(cfg.num_iter, desc="Optimising", disable=not verbose)
    for epoch in pbar:
        renderer.set_random_noise(epoch)
        optimizer.zero_grad_()

        # Forward
        sketch = renderer.get_image().to(cfg.device)
        losses = loss_fn(sketch, target_im.detach(), distractor_im, mode="train")
        loss = sum(losses.values())

        # Backward
        loss.backward()
        optimizer.step_()

        # Logging
        history["loss_train"].append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Save periodic snapshots
        if epoch % cfg.save_interval == 0:
            utils.plot_batch(
                target_im, sketch,
                os.path.join(cfg.output_dir, "jpg_logs"),
                epoch, title=f"iter{epoch}.jpg",
            )
            renderer.save_svg(os.path.join(cfg.output_dir, "svg_logs"), f"svg_iter{epoch}")

        # Evaluation
        if epoch % cfg.eval_interval == 0:
            with torch.no_grad():
                eval_losses = loss_fn(sketch, target_im, distractor_im, mode="eval")
                loss_eval = sum(eval_losses.values()).item()
                history["loss_eval"].append(loss_eval)

                cur_delta = loss_eval - best_loss
                if cur_delta < -min_delta:
                    best_loss = loss_eval
                    best_iter = epoch
                    terminate = False
                    utils.plot_batch(
                        target_im, sketch, cfg.output_dir,
                        epoch, title="best_iter.jpg",
                    )
                    renderer.save_svg(cfg.output_dir, "best_iter")
                elif abs(cur_delta) <= min_delta:
                    if terminate:
                        if verbose:
                            print(f"\n  Converged at epoch {epoch}.")
                        break
                    terminate = True
                else:
                    terminate = False

    # ── Save final results ────────────────────────────────────────

    renderer.save_svg(cfg.output_dir, "final_svg")
    best_svg = os.path.join(cfg.output_dir, "best_iter.svg")
    if os.path.exists(best_svg):
        utils.log_final(best_svg, cfg.device, best_iter, best_loss, cfg.output_dir)

    history["best_loss"] = best_loss
    history["best_iter"] = best_iter
    return history


# ─── High-level API ───────────────────────────────────────────────────

def run(
    target: str,
    distractor: Optional[str] = None,
    num_strokes: int = 16,
    num_iter: int = 1000,
    contrastive_weight: float = 0.5,
    output_dir: str = "outputs",
    image_scale: int = 224,
    width: float = 1.5,
    seed: int = 0,
    use_gpu: bool = True,
    mask_object: bool = False,
    fix_scale: bool = False,
    clip_model_name: str = "RN101",
    clip_fc_loss_weight: float = 0.1,
    num_aug_clip: int = 4,
    attention_init: bool = True,
    verbose: bool = True,
    **kwargs,
) -> Dict:
    """
    High-level function to generate a sketch.

    Args:
        target:              Path to target image.
        distractor:          Path to distractor image (None for normal mode).
        num_strokes:         Number of Bézier strokes.
        num_iter:            Optimisation iterations.
        contrastive_weight:  Weight for contrastive repulsion (0 = normal).
        output_dir:          Where to save outputs.
        image_scale:         Resolution for optimisation.
        width:               Stroke width.
        seed:                Random seed.
        use_gpu:             Use CUDA if available.
        mask_object:         Use U2Net to mask background.
        fix_scale:           Pad non-square images.
        clip_model_name:     CLIP backbone (RN101, RN50, ViT-B/32, etc.).
        clip_fc_loss_weight: FC layer loss weight.
        num_aug_clip:        Number of augmentations.
        attention_init:      Use attention for stroke placement.
        verbose:             Print progress.

    Returns:
        Dict with loss history and best results.
    """
    if distractor is None:
        contrastive_weight = 0.0

    cfg = SketchConfig(
        target=target,
        distractor=distractor,
        num_paths=num_strokes,
        num_iter=num_iter,
        contrastive_weight=contrastive_weight,
        output_dir=output_dir,
        image_scale=image_scale,
        width=width,
        seed=seed,
        use_gpu=use_gpu,
        mask_object=mask_object,
        fix_scale=fix_scale,
        clip_model_name=clip_model_name,
        clip_fc_loss_weight=clip_fc_loss_weight,
        num_aug_clip=num_aug_clip,
        attention_init=attention_init,
        **kwargs,
    )
    return train(cfg, verbose=verbose)


# ─── CLI entry point ──────────────────────────────────────────────────

def main():
    cfg = SketchConfig.from_cli()
    _setup_diffvg(cfg)
    history = train(cfg)
    # Save full config + history
    save_dict = {**asdict(cfg), **history}
    # Remove non-serialisable fields
    save_dict.pop("device", None)
    np.save(os.path.join(cfg.output_dir, "config.npy"), save_dict)


if __name__ == "__main__":
    main()