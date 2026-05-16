"""Simple spatiotemporal augmentations for fine-tuning.

Each transform takes (clip, target) where:
    clip:   (C, T, H, W) tensor in [0, 1]
    target: dict with 'cls' (C,), 'bbox' (C, 4) in normalized coords, 'angle' (C,) rad

Augmentations are designed to be OBB-aware: random horizontal flip negates
the angle and mirrors xc; small rotations rotate both image and angle.
"""

from __future__ import annotations

import math
import random
from typing import List

import torch
import torch.nn.functional as F


class Compose:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, clip, target):
        for t in self.transforms:
            clip, target = t(clip, target)
        return clip, target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, clip, target):
        if random.random() < self.p:
            clip = torch.flip(clip, dims=[-1])
            # xc -> 1 - xc; theta -> pi - theta (mod pi)
            target["bbox"][:, 0] = 1.0 - target["bbox"][:, 0]
            target["angle"] = (math.pi - target["angle"]) % math.pi
        return clip, target


class ColorJitter:
    """Per-clip color jitter (same params across frames)."""
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, clip, target):
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        # Only modify RGB channels (first 3) if depth present.
        rgb = clip[:3] * c
        rgb = rgb + (b - 1.0) * 0.5
        clip[:3] = rgb.clamp(0, 1)
        return clip, target


class Normalize:
    """ImageNet normalization for RGB channels; depth passes through."""
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __call__(self, clip, target):
        for i in range(3):
            clip[i] = (clip[i] - self.MEAN[i]) / self.STD[i]
        return clip, target


def default_train_transforms():
    return Compose([
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2),
        Normalize(),
    ])


def default_eval_transforms():
    return Compose([Normalize()])
