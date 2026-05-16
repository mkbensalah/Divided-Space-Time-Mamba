"""Unlabeled clip dataset for self-supervised pretraining.

Mixes multiple source directories (e.g. CHU-SJ recordings + synthetic clips
generated from public images) into a single dataset that yields short fixed-
length clips.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .chusj_dataset import _load_video_frames, _load_depth_frames


class PretrainClipDataset(Dataset):
    def __init__(
        self,
        video_dirs: List[str],
        num_frames: int = 16,
        img_size: int = 224,
        temporal_stride: int = 4,
        use_depth: bool = False,
        depth_suffix: str = "_depth",
    ):
        self.num_frames = num_frames
        self.img_size = img_size
        self.temporal_stride = temporal_stride
        self.use_depth = use_depth
        self.depth_suffix = depth_suffix

        self.videos: List[str] = []
        for d in video_dirs:
            d = Path(d)
            if not d.exists():
                continue
            for ext in ("mp4", "avi", "mov"):
                self.videos.extend(sorted(str(p) for p in d.rglob(f"*.{ext}")
                                          if depth_suffix not in p.stem))
        if not self.videos:
            raise RuntimeError(f"No videos found in {video_dirs}")

    def __len__(self):
        return len(self.videos)

    def _sample_indices(self, total: int) -> List[int]:
        span = self.temporal_stride * (self.num_frames - 1) + 1
        max_start = max(0, total - span)
        start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        return [start + i * self.temporal_stride for i in range(self.num_frames)]

    def __getitem__(self, idx):
        path = self.videos[idx]
        # Cheap total-frame estimate: assume 16-frame minimum; fall back if read fails.
        import cv2
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total < self.num_frames:
            indices = list(range(total)) + [total - 1] * (self.num_frames - total)
        else:
            indices = self._sample_indices(total)

        rgb = _load_video_frames(path, indices, self.img_size)        # (T, 3, H, W)

        if self.use_depth:
            depth_path = path.replace(".mp4", f"{self.depth_suffix}.mp4")
            if Path(depth_path).exists():
                depth = _load_depth_frames(depth_path, indices, self.img_size)
            else:
                depth = torch.zeros_like(rgb[:, :1])
            clip = torch.cat([rgb, depth], dim=1)
        else:
            clip = rgb

        clip = clip.permute(1, 0, 2, 3).contiguous()                  # (C, T, H, W)
        return {"clip": clip}


def pretrain_collate(batch):
    return {"clip": torch.stack([b["clip"] for b in batch])}
