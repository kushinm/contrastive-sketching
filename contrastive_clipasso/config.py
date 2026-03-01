"""Configuration for Contrastive CLIPasso."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class SketchConfig:
    """All parameters for a sketching run."""

    # --- Paths ---
    target: str = ""
    distractor: Optional[str] = None
    output_dir: str = "outputs"
    path_svg: str = "none"  # load existing SVG to continue from

    # --- Device ---
    use_gpu: bool = True
    seed: int = 0

    # --- Image preprocessing ---
    image_scale: int = 224
    mask_object: bool = False
    fix_scale: bool = False

    # --- Stroke parameters ---
    num_paths: int = 16
    width: float = 1.5
    control_points_per_seg: int = 4
    num_segments: int = 1
    num_stages: int = 1

    # --- Training ---
    num_iter: int = 1000
    lr: float = 1.0
    color_lr: float = 0.01
    lr_scheduler: bool = False
    save_interval: int = 50
    eval_interval: int = 50

    # --- CLIP loss ---
    clip_model_name: str = "RN101"
    clip_conv_loss: float = 1.0
    clip_conv_loss_type: str = "L2"
    clip_conv_layer_weights: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0, 0.0] ### zeroing out layer 3 too.
    )
    clip_fc_loss_weight: float = 0.1
    num_aug_clip: int = 4
    augmentations: str = "affine"

    # --- Contrastive ---
    contrastive_weight: float = 0.0  # 0 = normal mode, >0 = contrastive

    # --- Attention init ---
    attention_init: bool = True
    saliency_model: str = "clip"
    saliency_clip_model: str = "ViT-B/32"
    xdog_intersec: bool = True
    mask_object_attention: bool = False
    softmax_temp: float = 0.3
    text_target: str = "none"

    # --- Sparsity ---
    force_sparse: float = 0.0
    color_vars_threshold: float = 0.0
    noise_thresh: float = 0.5

    # --- Wandb ---
    use_wandb: bool = False
    wandb_user: str = ""
    wandb_project_name: str = "contrastive-clipasso"
    wandb_name: str = "run"

    # --- Display ---
    display_logs: bool = False

    # --- Derived (set in __post_init__) ---
    device: torch.device = field(default=None, init=False, repr=False)

    def __post_init__(self):
        set_seed(self.seed)

        # Resolve device
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Create output dirs
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "jpg_logs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "svg_logs"), exist_ok=True)

        # Infer contrastive mode
        if self.distractor is not None and self.contrastive_weight == 0.0:
            self.contrastive_weight = 0.5  # sensible default

    @property
    def is_contrastive(self) -> bool:
        return self.distractor is not None and self.contrastive_weight > 0

    @classmethod
    def from_cli(cls) -> "SketchConfig":
        """Build config from command-line arguments."""
        import argparse

        p = argparse.ArgumentParser(description="Contrastive CLIPasso")
        p.add_argument("target", help="Path to target image")
        p.add_argument("--distractor", type=str, default=None,
                        help="Path to distractor image (enables contrastive mode)")
        p.add_argument("--output_dir", type=str, default="outputs")
        p.add_argument("--num_strokes", type=int, default=16)
        p.add_argument("--num_iter", type=int, default=1000)
        p.add_argument("--contrastive_weight", type=float, default=0.0)
        p.add_argument("--image_scale", type=int, default=224)
        p.add_argument("--width", type=float, default=1.5)
        p.add_argument("--lr", type=float, default=1.0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--use_gpu", action="store_true", default=True)
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--mask_object", action="store_true")
        p.add_argument("--fix_scale", action="store_true")
        p.add_argument("--num_segments", type=int, default=1)
        p.add_argument("--save_interval", type=int, default=50)
        p.add_argument("--eval_interval", type=int, default=50)
        p.add_argument("--clip_model_name", type=str, default="RN101")
        p.add_argument("--clip_conv_layer_weights", type=str, default="0,0,1.0,1.0,0")
        p.add_argument("--clip_fc_loss_weight", type=float, default=0.1)
        p.add_argument("--num_aug_clip", type=int, default=4)
        p.add_argument("--attention_init", action="store_true", default=True)
        p.add_argument("--no_attention_init", action="store_true")
        p.add_argument("--saliency_model", type=str, default="clip")
        p.add_argument("--force_sparse", type=float, default=0.0)
        p.add_argument("--use_wandb", action="store_true")
        p.add_argument("--wandb_name", type=str, default="run")

        args = p.parse_args()

        weights = [float(x) for x in args.clip_conv_layer_weights.split(",")]
        use_gpu = args.use_gpu and not args.cpu
        attention_init = args.attention_init and not args.no_attention_init

        return cls(
            target=args.target,
            distractor=args.distractor,
            output_dir=args.output_dir,
            num_paths=args.num_strokes,
            num_iter=args.num_iter,
            contrastive_weight=args.contrastive_weight,
            image_scale=args.image_scale,
            width=args.width,
            lr=args.lr,
            seed=args.seed,
            use_gpu=use_gpu,
            mask_object=args.mask_object,
            fix_scale=args.fix_scale,
            num_segments=args.num_segments,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            clip_model_name=args.clip_model_name,
            clip_conv_layer_weights=weights,
            clip_fc_loss_weight=args.clip_fc_loss_weight,
            num_aug_clip=args.num_aug_clip,
            attention_init=attention_init,
            saliency_model=args.saliency_model,
            force_sparse=args.force_sparse,
            use_wandb=args.use_wandb,
            wandb_name=args.wandb_name,
        )


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
