"""
CLIP-based losses for sketch optimisation.

Normal mode:   minimise dist(sketch, target)
Contrastive:   minimise dist(sketch, target) − λ · dist(sketch, distractor)
"""

from __future__ import annotations

import collections
from typing import Dict, Optional

import clip
import torch
import torch.nn as nn
from torchvision import models, transforms


# ─── Distance metrics ────────────────────────────────────────────────

def l2_layers(xs, ys, _model_name):
    return [torch.square(x - y).mean() for x, y in zip(xs, ys)]


def l1_layers(xs, ys, _model_name):
    return [torch.abs(x - y).mean() for x, y in zip(xs, ys)]


def cos_layers(xs, ys, model_name):
    if "RN" in model_name:
        # For ResNet features (spatial maps), flatten and use L2
        return [torch.square(x - y).mean() for x, y in zip(xs, ys)]
    return [(1 - torch.cosine_similarity(x, y, dim=1)).mean() for x, y in zip(xs, ys)]


DISTANCE_FNS = {"L2": l2_layers, "L1": l1_layers, "Cos": cos_layers}


# ─── Visual encoder (ViT hook-based) ─────────────────────────────────

class CLIPVisualEncoder(nn.Module):
    """Extracts intermediate features from a CLIP ViT model via hooks."""

    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None
        self.num_layers = len(clip_model.visual.transformer.resblocks)
        for i in range(self.num_layers):
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self._make_hook(i)
            )

    def _make_hook(self, name):
        def hook(_module, _input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(1, 0, 2)  # LND → NLD
            else:
                self.featuremaps[name] = output
        return hook

    def forward(self, x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(self.num_layers)]
        return fc_features, featuremaps


# ─── Main loss class ──────────────────────────────────────────────────

class ContrastiveCLIPLoss(nn.Module):
    """
    CLIP-based loss that supports both normal and contrastive sketching.

    Normal mode (distractor_im=None):
        loss = conv_loss(sketch, target) + fc_loss(sketch, target)

    Contrastive mode (distractor_im provided):
        loss = conv_loss(sketch, target) + fc_loss(sketch, target)
             − λ · [conv_loss(sketch, distractor) + fc_loss(sketch, distractor)]

    The contrastive term encourages the sketch to emphasise features
    that distinguish the target from the distractor.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device

        # Load CLIP model
        self.clip_model_name = cfg.clip_model_name
        self.model, clip_preprocess = clip.load(
            self.clip_model_name, self.device, jit=False
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Set up feature extractor
        if self.clip_model_name.startswith("ViT"):
            self.visual_encoder = CLIPVisualEncoder(self.model)
        else:
            self.visual_model = self.model.visual
            layers = list(self.model.visual.children())
            self.layer1 = layers[8]
            self.layer2 = layers[9]
            self.layer3 = layers[10]
            self.layer4 = layers[11]
            self.att_pool2d = layers[12]

        # Transforms
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        # Augmentation pipeline
        aug_list = []
        if "affine" in cfg.augmentations:
            aug_list.append(transforms.RandomPerspective(fill=0, p=1.0, distortion_scale=0.5))
            aug_list.append(transforms.RandomResizedCrop(224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        aug_list.append(
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            )
        )
        self.augment_trans = transforms.Compose(aug_list)

        # Distance metric
        self.conv_loss_type = cfg.clip_conv_loss_type
        self.distance_fn = DISTANCE_FNS[self.conv_loss_type]

        # Weights — auto-pad to match the number of conv feature stages.
        # ResNets produce 5 stages (stem + layers 1-4).
        # ViTs produce N stages (one per transformer block: 12 for ViT-B, 24 for ViT-L).
        if self.clip_model_name.startswith("ViT"):
            num_conv_stages = self.visual_encoder.num_layers
        else:
            num_conv_stages = 5  # stem, layer1, layer2, layer3, layer4
        
        self.layer_weights = list(cfg.clip_conv_layer_weights)
        if len(self.layer_weights) < num_conv_stages:
            # Pad with zeros — only the explicitly specified layers contribute
            self.layer_weights += [0.0] * (num_conv_stages - len(self.layer_weights))
        elif len(self.layer_weights) > num_conv_stages:
            # Truncate if user provided too many
            self.layer_weights = self.layer_weights[:num_conv_stages]
        self.fc_loss_weight = cfg.clip_fc_loss_weight
        self.contrastive_weight = cfg.contrastive_weight
        self.num_augs = cfg.num_aug_clip

    # ---- feature extraction ----

    def _extract_features(self, images):
        """Extract fc + conv features from images."""
        if self.clip_model_name.startswith("RN"):
            return self._forward_resnet(images)
        else:
            return self.visual_encoder(images)

    def _forward_resnet(self, x):
        """Forward pass through CLIP ResNet, returning fc + intermediate features."""
        def stem(m, x):
            for conv, bn, relu in [(m.conv1, m.bn1, m.relu1), (m.conv2, m.bn2, m.relu2), (m.conv3, m.bn3, m.relu3)]:
                x = relu(bn(conv(x)))
            x = m.avgpool(x)
            return x
        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y, [x, x1, x2, x3, x4]

    # ---- augmentation ----

    def _augment_pair(self, sketch, reference, mode):
        """Produce augmented batches of sketch and reference image."""
        xs = [self.normalize_transform(sketch)]
        ys = [self.normalize_transform(reference)]
        if mode == "train":
            for _ in range(self.num_augs):
                pair = self.augment_trans(torch.cat([sketch, reference]))
                xs.append(pair[0].unsqueeze(0))
                ys.append(pair[1].unsqueeze(0))
        return torch.cat(xs, dim=0).to(self.device), torch.cat(ys, dim=0).to(self.device)

    # ---- pairwise loss computation ----

    def _compute_pair_loss(self, sketch, reference, mode="train") -> Dict[str, torch.Tensor]:
        """Compute conv + fc loss between sketch and a single reference image."""
        xs, ys = self._augment_pair(sketch, reference, mode)
        xs_fc, xs_conv = self._extract_features(xs.contiguous())
        ys_fc, ys_conv = self._extract_features(ys.detach())

        conv_losses = self.distance_fn(xs_conv, ys_conv, self.clip_model_name)

        loss_dict = {}
        for layer_idx, w in enumerate(self.layer_weights):
            if w > 0:
                loss_dict[f"conv_L{layer_idx}"] = conv_losses[layer_idx] * w

        if self.fc_loss_weight > 0:
            fc_loss = (1 - torch.cosine_similarity(xs_fc, ys_fc, dim=1)).mean()
            loss_dict["fc"] = fc_loss * self.fc_loss_weight

        return loss_dict

    # ---- main forward ----

    def forward(
        self,
        sketch: torch.Tensor,
        target: torch.Tensor,
        distractor: Optional[torch.Tensor] = None,
        mode: str = "train",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss.

        Args:
            sketch:     rendered sketch [1, C, H, W]
            target:     target image    [1, C, H, W]
            distractor: distractor image [1, C, H, W] or None
            mode:       "train" (with augmentation) or "eval" (without)

        Returns:
            Dict of named loss components. Sum them to get total loss.
        """
        # Attraction loss: pull sketch toward target
        attract_dict = self._compute_pair_loss(sketch, target, mode)
        result = {f"attract_{k}": v for k, v in attract_dict.items()}

        # Contrastive repulsion loss: push sketch away from distractor
        if distractor is not None and self.contrastive_weight > 0:
            repel_dict = self._compute_pair_loss(sketch, distractor, mode)
            for k, v in repel_dict.items():
                # Negative sign: we want to MAXIMISE distance from distractor
                result[f"repel_{k}"] = -self.contrastive_weight * v

        return result