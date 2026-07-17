"""Utility functions for image loading, masking, SVG rendering, and visualisation."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pydiffvg
import torch
from PIL import Image
from skimage.transform import resize
from torchvision import transforms
from torchvision.utils import make_grid

from . import compat  # noqa: F401 — patches numpy aliases on import
from .compat import BICUBIC


# ─── Image loading ────────────────────────────────────────────────────

def load_target(cfg) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Load and preprocess the target image, returning (image_tensor, mask)."""
    target = Image.open(cfg.target)
    if target.mode == "RGBA":
        bg = Image.new("RGBA", target.size, "WHITE")
        bg.paste(target, (0, 0), target)
        target = bg
    target = target.convert("RGB")

    mask = None
    if cfg.mask_object:
        masked_im, mask = get_mask_u2net(cfg, target)
        target = masked_im

    if cfg.fix_scale:
        target = fix_image_scale(target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(
            transforms.Resize((cfg.image_scale, cfg.image_scale), interpolation=BICUBIC)
        )
    else:
        transforms_.append(transforms.Resize(cfg.image_scale, interpolation=BICUBIC))
        transforms_.append(transforms.CenterCrop(cfg.image_scale))
    transforms_.append(transforms.ToTensor())

    data_transforms = transforms.Compose(transforms_)
    target_tensor = data_transforms(target).unsqueeze(0).to(cfg.device)
    return target_tensor, mask


def load_distractor(cfg) -> torch.Tensor:
    """Load and preprocess the distractor image."""
    distractor = Image.open(cfg.distractor)
    if distractor.mode == "RGBA":
        bg = Image.new("RGBA", distractor.size, "WHITE")
        bg.paste(distractor, (0, 0), distractor)
        distractor = bg
    distractor = distractor.convert("RGB")

    if cfg.fix_scale:
        distractor = fix_image_scale(distractor)

    transforms_ = []
    if distractor.size[0] != distractor.size[1]:
        transforms_.append(
            transforms.Resize((cfg.image_scale, cfg.image_scale), interpolation=BICUBIC)
        )
    else:
        transforms_.append(transforms.Resize(cfg.image_scale, interpolation=BICUBIC))
        transforms_.append(transforms.CenterCrop(cfg.image_scale))
    transforms_.append(transforms.ToTensor())

    data_transforms = transforms.Compose(transforms_)
    return data_transforms(distractor).unsqueeze(0).to(cfg.device)


# ─── Image scale fix ──────────────────────────────────────────────────

def fix_image_scale(im: Image.Image) -> Image.Image:
    """Pad a non-square image onto a white square background."""
    im_np = np.array(im) / 255.0
    h, w = im_np.shape[:2]
    max_len = max(h, w) + 20
    bg = np.ones((max_len, max_len, 3))
    y, x = max_len // 2 - h // 2, max_len // 2 - w // 2
    bg[y : y + h, x : x + w] = im_np
    bg = (bg / bg.max() * 255).astype(np.uint8)
    return Image.fromarray(bg)


# ─── U2Net masking ────────────────────────────────────────────────────

def get_mask_u2net(cfg, pil_im: Image.Image):
    """Use U2Net to generate a foreground mask."""
    # Import U2Net — resolve path robustly regardless of working directory
    import sys
    # Try multiple possible locations for U2Net_
    u2net_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "U2Net_"),
        os.path.join(os.getcwd(), "U2Net_"),
        os.path.join(os.path.dirname(os.path.abspath(cfg.target)), "..", "U2Net_"),
    ]
    u2net_path = None
    for candidate in u2net_candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "model.py")):
            u2net_path = candidate
            break
    
    if u2net_path is None:
        raise FileNotFoundError(
            "Could not find U2Net_/model.py. Expected locations:\n"
            + "\n".join(f"  - {os.path.abspath(c)}" for c in u2net_candidates)
            + "\n\nMake sure U2Net_ directory with model.py exists in the project root."
        )
    
    if u2net_path not in sys.path:
        sys.path.insert(0, u2net_path)
    from model import U2NET

    w, h = pil_im.size
    im_size = min(w, h)
    data_transforms = transforms.Compose([
        transforms.Resize(min(320, im_size), interpolation=BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    input_tensor = data_transforms(pil_im).unsqueeze(0).to(cfg.device)
    model_dir = os.path.join(u2net_path, "saved_models", "u2net.pth")
    net = U2NET(3, 1)
    if torch.cuda.is_available() and cfg.use_gpu:
        net.load_state_dict(torch.load(model_dir))
        net.to(cfg.device)
    else:
        net.load_state_dict(torch.load(model_dir, map_location="cpu"))
    net.eval()

    with torch.no_grad():
        d1, *_ = net(input_tensor.detach())
    pred = d1[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    predict = pred.clone()
    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1

    mask = torch.cat([predict, predict, predict], dim=0).permute(1, 2, 0).cpu().numpy()
    mask = resize(mask, (h, w), anti_aliasing=False)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    im_np = np.array(pil_im).astype(float) / 255.0
    im_np = mask * im_np
    im_np[mask == 0] = 1
    im_final = (im_np * 255).astype(np.uint8)
    im_final = Image.fromarray(im_final)

    return im_final, predict


# ─── SVG reading ──────────────────────────────────────────────────────

def read_svg(path_svg: str, device=None, multiply: bool = False) -> torch.Tensor:
    """Render an SVG file to a [H, W, 3] tensor."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(path_svg)
    if multiply:
        canvas_width *= 2
        canvas_height *= 2
        for path in shapes:
            path.points *= 2
            path.stroke_width *= 2

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    img = pydiffvg.RenderFunction.apply(
        canvas_width, canvas_height, 2, 2, 0, None, *scene_args
    )
    # Composite on whatever device diffvg rendered to (CPU if built without CUDA)
    render_dev = img.device
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
        img.shape[0], img.shape[1], 3, device=render_dev
    ) * (1 - img[:, :, 3:4])
    return img[:, :, :3]


# ─── Plotting / logging ──────────────────────────────────────────────

def plot_batch(inputs, outputs, output_dir, step, title="iter.jpg", use_wandb=False):
    """Save a side-by-side comparison of input and output."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    grid_in = make_grid(inputs.clone().detach(), normalize=True, pad_value=2)
    ax1.imshow(np.transpose(grid_in.cpu().numpy(), (1, 2, 0)))
    ax1.set_title("Input")
    ax1.axis("off")

    grid_out = make_grid(outputs, normalize=False, pad_value=2)
    ax2.imshow(np.transpose(grid_out.detach().cpu().numpy(), (1, 2, 0)))
    ax2.set_title("Sketch")
    ax2.axis("off")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, title), dpi=100)
    plt.close()


def save_input_image(inputs, output_dir):
    """Save a copy of the input image to the output directory."""
    inp = inputs[0].cpu().clone().detach().permute(1, 2, 0).numpy()
    inp = (inp - inp.min()) / (inp.max() - inp.min())
    inp = (inp * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(output_dir, "input.png"), inp)


def log_final(path_svg, device, best_iter, best_loss, output_dir):
    """Render the best SVG and save as final result."""
    img = read_svg(path_svg, device, multiply=True)
    result = Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))
    result.save(os.path.join(output_dir, "final_sketch.png"))
    print(f"  Best iteration: {best_iter}, loss: {best_loss:.4f}")
    print(f"  Final sketch saved to {output_dir}/final_sketch.png")
    print(f"  SVG saved to {path_svg}")