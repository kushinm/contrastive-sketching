"""
Stroke-based image renderer using differentiable SVG (pydiffvg).

Handles stroke initialisation (random or attention-guided), rendering, and parameter access.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import clip
import numpy as np
import pydiffvg
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torchvision import transforms


class Painter(nn.Module):
    """Differentiable SVG renderer that optimises Bézier stroke positions."""

    def __init__(self, cfg, target_im: torch.Tensor, mask=None):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        # render_device is where diffvg actually renders — CPU when diffvg
        # was built without CUDA (common on CUDA 12). Shape points must
        # live here. The final image gets moved to self.device for CLIP.
        self.render_device = (
            cfg.device if pydiffvg.get_use_gpu() else torch.device("cpu")
        )
        self.num_paths = cfg.num_paths
        self.num_segments = cfg.num_segments
        self.width = cfg.width
        self.control_points_per_seg = cfg.control_points_per_seg
        self.opacity_optim = cfg.force_sparse > 0
        self.num_stages = cfg.num_stages
        self.add_random_noise = "noise" in cfg.augmentations
        self.noise_thresh = cfg.noise_thresh
        self.softmax_temp = cfg.softmax_temp

        self.canvas_width = cfg.image_scale
        self.canvas_height = cfg.image_scale

        self.shapes: List = []
        self.shape_groups: List = []
        self.points_vars: List = []
        self.color_vars: List = []
        self.optimize_flag: List[bool] = []
        self.strokes_counter = 0

        self.path_svg = cfg.path_svg
        self.strokes_per_stage = self.num_paths

        # Attention-based initialisation
        self.attention_init = cfg.attention_init
        self.target_path = cfg.target
        self.saliency_model = cfg.saliency_model
        self.xdog_intersec = cfg.xdog_intersec
        self.mask_object_attention = cfg.mask_object_attention
        self.text_target = cfg.text_target
        self.saliency_clip_model = cfg.saliency_clip_model

        self._define_attention_input(target_im)
        self.mask = mask
        self.attention_map = self._compute_attention() if self.attention_init else None
        self.thresh = self._compute_threshold_map() if self.attention_init else None

    # ─── Image init / rendering ───────────────────────────────────────

    def init_image(self, stage: int = 0) -> torch.Tensor:
        """Create initial strokes and render the first image."""
        if stage > 0:
            self.optimize_flag = [False] * len(self.shapes)
            for _ in range(self.strokes_per_stage):
                self._add_stroke()
                self.optimize_flag.append(True)
        else:
            num_existing = 0
            if self.path_svg != "none":
                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = (
                    pydiffvg.svg_to_scene(self.path_svg)
                )
                num_existing = len(self.shapes)
            for _ in range(num_existing, self.num_paths):
                self._add_stroke()
            self.optimize_flag = [True] * len(self.shapes)
        return self.get_image()

    def get_image(self) -> torch.Tensor:
        """Render current strokes to a [1, 3, H, W] image tensor on self.device."""
        img = self._render()  # on render_device (may be CPU)
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=self.render_device
        ) * (1 - opacity)
        img = img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2)
        return img.to(self.device)  # move to CUDA for CLIP loss

    def _render(self) -> torch.Tensor:
        """Raw render via pydiffvg, with automatic CPU fallback."""
        if self.opacity_optim:
            for group in self.shape_groups:
                group.stroke_color.data[:3].clamp_(0.0, 0.0)
                group.stroke_color.data[-1].clamp_(0.0, 1.0)

        if self.add_random_noise:
            if random.random() > self.noise_thresh:
                eps = 0.01 * min(self.canvas_width, self.canvas_height)
                for path in self.shapes:
                    path.points.data.add_(eps * torch.randn_like(path.points))

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups
        )
        try:
            img = pydiffvg.RenderFunction.apply(
                self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args
            )
        except RuntimeError as e:
            if "not compiled with GPU" in str(e):
                # diffvg wasn't built with CUDA — fall back to CPU rendering.
                # This is the normal case on CUDA 12 systems.
                print("  diffvg GPU unavailable, switching to CPU rendering.")
                pydiffvg.set_use_gpu(False)
                pydiffvg.set_device(torch.device("cpu"))
                self.render_device = torch.device("cpu")
                # Move all shape data to CPU
                for path in self.shapes:
                    path.points = path.points.to("cpu")
                    if path.stroke_width is not None:
                        path.stroke_width = path.stroke_width.to("cpu") if isinstance(path.stroke_width, torch.Tensor) else path.stroke_width
                for group in self.shape_groups:
                    if group.stroke_color is not None:
                        group.stroke_color = group.stroke_color.to("cpu")
                    if group.fill_color is not None:
                        group.fill_color = group.fill_color.to("cpu")
                # Re-serialize and retry on CPU
                scene_args = pydiffvg.RenderFunction.serialize_scene(
                    self.canvas_width, self.canvas_height, self.shapes, self.shape_groups
                )
                img = pydiffvg.RenderFunction.apply(
                    self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args
                )
            else:
                raise
        return img

    def _add_stroke(self):
        """Add a single Bézier stroke."""
        path = self._make_path()
        self.shapes.append(path)
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(self.shapes) - 1]),
            fill_color=None,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        self.shape_groups.append(group)

    def _make_path(self) -> pydiffvg.Path:
        """Create a Bézier path with attention or random initialisation."""
        num_ctrl = torch.zeros(self.num_segments, dtype=torch.int32) + (
            self.control_points_per_seg - 2
        )
        if self.attention_init:
            p0 = self.inds_normalised[self.strokes_counter]
        else:
            p0 = (random.random(), random.random())

        points = [p0]
        for _ in range(self.num_segments):
            radius = 0.05
            for _ in range(self.control_points_per_seg - 1):
                p1 = (
                    p0[0] + radius * (random.random() - 0.5),
                    p0[1] + radius * (random.random() - 0.5),
                )
                points.append(p1)
                p0 = p1

        points = torch.tensor(points).to(self.render_device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height

        path = pydiffvg.Path(
            num_control_points=num_ctrl,
            points=points,
            stroke_width=torch.tensor(self.width),
            is_closed=False,
        )
        self.strokes_counter += 1
        return path

    # ─── Parameter access ─────────────────────────────────────────────

    def parameters(self):
        """Point parameters to optimise."""
        self.points_vars = []
        for i, path in enumerate(self.shapes):
            if self.optimize_flag[i]:
                path.points.requires_grad = True
                self.points_vars.append(path.points)
        return self.points_vars

    def get_points_params(self):
        return self.points_vars

    def set_color_parameters(self):
        self.color_vars = []
        for i, group in enumerate(self.shape_groups):
            if self.optimize_flag[i]:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)
        return self.color_vars

    def get_color_parameters(self):
        return self.color_vars

    def set_random_noise(self, epoch: int):
        if epoch % self.cfg.save_interval == 0:
            self.add_random_noise = False
        else:
            self.add_random_noise = "noise" in self.cfg.augmentations

    # ─── SVG I/O ──────────────────────────────────────────────────────

    def save_svg(self, output_dir: str, name: str):
        pydiffvg.save_svg(
            f"{output_dir}/{name}.svg",
            self.canvas_width,
            self.canvas_height,
            self.shapes,
            self.shape_groups,
        )

    # ─── Attention / saliency ─────────────────────────────────────────

    def _define_attention_input(self, target_im: torch.Tensor):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval()
        self.image_input_attn_clip = preprocess.transforms[-1](target_im).to(self.device)
        del model

    def _compute_attention(self):
        if self.saliency_model == "dino":
            return self._dino_attention()
        else:
            return self._clip_attention()

    def _compute_threshold_map(self):
        if self.saliency_model == "dino":
            return self._set_inds_dino()
        else:
            return self._set_inds_clip()

    def _clip_attention(self):
        model, _preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval()

        if "RN" in self.saliency_clip_model:
            text_input = clip.tokenize([self.text_target]).to(self.device)
            saliency_layer = "layer4"
            attn_map = _gradCAM(
                model.visual,
                self.image_input_attn_clip,
                model.encode_text(text_input).float(),
                getattr(model.visual, saliency_layer),
            )
            attn_map = attn_map.squeeze().detach().cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        else:
            text_input = clip.tokenize([self.text_target]).to(self.device)
            attn_map = _interpret(self.image_input_attn_clip, text_input, model, self.device)

        del model
        return attn_map

    def _dino_attention(self):
        patch_size = 8
        threshold = 0.6
        mean_in = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(self.device)
        std_in = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(self.device)
        totens = transforms.Compose([
            transforms.Resize((self.canvas_height, self.canvas_width)),
            transforms.ToTensor(),
        ])
        dino_model = torch.hub.load("facebookresearch/dino:main", "dino_vits8").eval().to(self.device)
        main_im = Image.open(self.target_path).convert("RGB")
        main_im_tensor = totens(main_im).to(self.device)
        img = (main_im_tensor.unsqueeze(0) - mean_in) / std_in
        w_feat = img.shape[-2] // patch_size
        h_feat = img.shape[-1] // patch_size

        with torch.no_grad():
            attn = dino_model.get_last_selfattention(img).detach().cpu()[0]

        nh = attn.shape[0]
        attn = attn[:, 0, 1:].reshape(nh, -1)
        attn = attn.reshape(nh, w_feat, h_feat).float()
        attn = F.interpolate(attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()
        del dino_model
        return attn

    def _softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum()

    def _set_inds_clip(self):
        attn_map = (self.attention_map - self.attention_map.min()) / (
            self.attention_map.max() - self.attention_map.min()
        )
        if self.xdog_intersec:
            xdog = _XDoG()
            im_xdog = xdog(self.image_input_attn_clip[0].permute(1, 2, 0).cpu().numpy(), k=10)
            attn_map = (1 - im_xdog) * attn_map

        attn_soft = np.copy(attn_map)
        attn_soft[attn_map > 0] = self._softmax(attn_map[attn_map > 0], tau=self.softmax_temp)

        k = self.num_stages * self.num_paths
        inds = np.random.choice(
            range(attn_map.flatten().shape[0]), size=k, replace=False, p=attn_soft.flatten()
        )
        inds = np.array(np.unravel_index(inds, attn_map.shape)).T
        self.inds = inds
        self.inds_normalised = np.zeros(inds.shape)
        self.inds_normalised[:, 0] = inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] = inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()
        return attn_soft

    def _set_inds_dino(self):
        k = max(3, (self.num_stages * self.num_paths) // 6 + 1)
        num_heads = self.attention_map.shape[0]
        self.inds = np.zeros((k * num_heads, 2))
        softmax_fn = nn.Softmax(dim=1)

        for i in range(num_heads):
            topk, _ = np.unique(self.attention_map[i].numpy(), return_index=True)
            topk = topk[::-1][:k]
            cur = self.attention_map[i].numpy()
            prob = cur.flatten()
            prob[prob > topk[-1]] = 1
            prob[prob <= topk[-1]] = 0
            prob = prob / prob.sum()
            chosen = np.random.choice(range(cur.flatten().shape[0]), size=k, replace=False, p=prob)
            chosen = np.unravel_index(chosen, cur.shape)
            self.inds[i * k : i * k + k, 0] = chosen[0]
            self.inds[i * k : i * k + k, 1] = chosen[1]

        sum_attn = self.attention_map.sum(0).numpy()
        sum_attn = sum_attn / sum_attn.sum()
        prob_sum = sum_attn[self.inds[:, 0].astype(int), self.inds[:, 1].astype(int)]
        prob_sum = prob_sum / prob_sum.sum()
        new_inds = []
        for _ in range(self.num_stages):
            new_inds.extend(
                np.random.choice(range(self.inds.shape[0]), size=self.num_paths, replace=False, p=prob_sum)
            )
        self.inds = self.inds[new_inds]
        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] = self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] = self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()
        return None  # dino threshold not needed for rendering

    # ─── Accessors for visualisation ──────────────────────────────────

    def get_attn(self):
        return self.attention_map

    def get_thresh(self):
        return self.thresh

    def get_inds(self):
        return self.inds


class PainterOptimizer:
    """Wraps Adam optimiser(s) for stroke points and optionally colors."""

    def __init__(self, cfg, renderer: Painter):
        self.renderer = renderer
        self.points_lr = cfg.lr
        self.color_lr = cfg.color_lr
        self.optim_color = cfg.force_sparse > 0
        self.points_optim = None
        self.color_optim = None

    def init_optimizers(self):
        self.points_optim = torch.optim.Adam(self.renderer.parameters(), lr=self.points_lr)
        if self.optim_color:
            self.color_optim = torch.optim.Adam(
                self.renderer.set_color_parameters(), lr=self.color_lr
            )

    def zero_grad_(self):
        self.points_optim.zero_grad()
        if self.optim_color and self.color_optim is not None:
            self.color_optim.zero_grad()

    def step_(self):
        self.points_optim.step()
        if self.optim_color and self.color_optim is not None:
            self.color_optim.step()

    def get_lr(self):
        return self.points_optim.param_groups[0]["lr"]


# ─── Helper functions (from original CLIPasso) ────────────────────────

class _Hook:
    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self._save)

    def _save(self, _module, _input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.hook.remove()

    @property
    def activation(self):
        return self.data

    @property
    def gradient(self):
        return self.data.grad


def _patch_clip_attn_probs(model):
    """
    Monkey-patch a CLIP ViT model so each ResidualAttentionBlock stores
    `attn_probs` after its forward pass.

    The original CLIPasso used a local CLIP fork (CLIP_/) that had this built in.
    The standard OpenAI CLIP package does not store attention weights, so we
    replace each block's `attention()` method with one that passes
    `need_weights=True` and saves the result.
    """
    for block in model.visual.transformer.resblocks:
        original_attn_module = block.attn  # nn.MultiheadAttention

        def make_patched_attention(blk, attn_module):
            def patched_attention(x):
                # Replicate the original CLIP attention() method but with need_weights=True
                attn_mask = blk.attn_mask
                if attn_mask is not None:
                    attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)
                out, attn_weights = attn_module(
                    x, x, x, need_weights=True, attn_mask=attn_mask
                )
                blk.attn_probs = attn_weights  # store on the block for _interpret
                return out
            return patched_attention

        block.attention = make_patched_attention(block, original_attn_module)


def _interpret(image, texts, model, device):
    _patch_clip_attn_probs(model)
    images = image.repeat(1, 1, 1, 1)
    model.encode_image(images)
    model.zero_grad()
    blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
    for blk in blocks:
        cam = blk.attn_probs.detach()
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)

    cams = torch.cat(
        [blk.attn_probs.detach().reshape(1, -1, blk.attn_probs.shape[-1], blk.attn_probs.shape[-1]).clamp(min=0).mean(dim=1) for blk in blocks]
    )
    cams_avg = cams[:, 0, 1:]
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0).reshape(1, 1, 7, 7)
    image_relevance = F.interpolate(image_relevance, size=224, mode="bicubic")
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (
        image_relevance.max() - image_relevance.min()
    )
    return image_relevance


def _gradCAM(model, input_tensor, target, layer):
    if input_tensor.grad is not None:
        input_tensor.grad.data.zero_()
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    with _Hook(layer) as hook:
        output = model(input_tensor)
        output.backward(target)
        grad = hook.gradient.float()
        act = hook.activation.float()
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        gradcam = torch.clamp(gradcam, min=0)

    gradcam = F.interpolate(gradcam, input_tensor.shape[2:], mode="bicubic", align_corners=False)
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
    return gradcam


class _XDoG:
    def __init__(self):
        self.gamma = 0.98
        self.phi = 200
        self.eps = -0.1
        self.sigma = 0.8

    def __call__(self, im, k=10):
        if im.shape[2] == 3:
            im = rgb2gray(im)
        imf1 = gaussian_filter(im, self.sigma)
        imf2 = gaussian_filter(im, self.sigma * k)
        imdiff = imf1 - self.gamma * imf2
        imdiff = (imdiff < self.eps) * 1.0 + (imdiff >= self.eps) * (1.0 + np.tanh(self.phi * imdiff))
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()
        th = threshold_otsu(imdiff)
        imdiff = (imdiff >= th).astype("float32")
        return imdiff