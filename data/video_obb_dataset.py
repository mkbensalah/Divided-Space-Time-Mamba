"""General-purpose video dataset for oriented bounding box (OBB) detection.

Works with any
annotated video collection — clinical, surveillance, sports, robotics, etc.

Directory layout
----------------
    root/
        annotations/
            {video_id}.json     one file per video (schema below)
        splits/
            train.txt           one video_id per line
            val.txt
            test.txt            (any split name you like)

If your videos live outside the root, set an absolute "file" key in each
annotation JSON.

Annotation JSON schema (one file per video)
-------------------------------------------
    {
        "video_id": "vid_001",          // optional, falls back to filename stem
        "file":     "videos/vid_001.mp4", // path relative to root OR absolute
        "depth_file": "videos/vid_001_depth.mp4",  // optional, RGB-D
        "fps": 30,
        "height": 720,
        "width":  1280,
        "frames": [
            {
                "index": 0,
                "objects": [
                    {
                        "class":     "person",
                        "xc":        320.5,   // pixel coords at native resolution
                        "yc":        240.3,
                        "w":         80.0,
                        "h":        120.0,
                        "theta_deg": 5.0      // CCW rotation of the long axis, [0, 180)
                    }
                ]
            }
        ]
    }

If "file" is absent the video is assumed to be at videos/{video_id}.mp4 under root.

Per-clip target format
----------------------
Each __getitem__ returns a clip tensor of shape (C, T, H, W) and a target dict:
    cls   : (num_classes,)    float — 1 if class is present in this clip
    bbox  : (num_classes, 4) float — (xc, yc, w, h) normalised to [0, 1]
    angle : (num_classes,)   float — angle in radians, [0, π)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import decord
    decord.bridge.set_bridge("torch")
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False

import cv2


def _load_video_frames(path: str, indices: List[int], target_size: int) -> torch.Tensor:
    """Load specified frame indices from a video file, resized to square target_size."""
    if _HAS_DECORD:
        import decord as _decord
        vr = _decord.VideoReader(path)
        frames = vr.get_batch(indices).permute(0, 3, 1, 2).float() / 255.0
    else:
        idx_set = set(indices)
        cap = cv2.VideoCapture(path)
        frames_dict: Dict[int, np.ndarray] = {}
        i = 0
        while i <= max(indices):
            ok, frame = cap.read()
            if not ok:
                break
            if i in idx_set:
                frames_dict[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()
        arr = np.stack([frames_dict[i] for i in indices], axis=0)
        frames = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    return torch.nn.functional.interpolate(
        frames, size=(target_size, target_size), mode="bilinear", align_corners=False
    )


def _load_depth_frames(path: str, indices: List[int], target_size: int) -> torch.Tensor:
    """Load depth frames (single channel) from a video file."""
    if _HAS_DECORD:
        import decord as _decord
        vr = _decord.VideoReader(path)
        frames = vr.get_batch(indices).float()[..., :1].permute(0, 3, 1, 2) / 255.0
    else:
        idx_set = set(indices)
        cap = cv2.VideoCapture(path)
        frames_dict: Dict[int, np.ndarray] = {}
        i = 0
        while i <= max(indices):
            ok, frame = cap.read()
            if not ok:
                break
            if i in idx_set:
                frames_dict[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            i += 1
        cap.release()
        arr = np.stack([frames_dict[i] for i in indices], axis=0)[..., None]
        frames = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    return torch.nn.functional.interpolate(
        frames, size=(target_size, target_size), mode="bilinear", align_corners=False
    )


class VideoOBBDataset(Dataset):
    """Video dataset for OBB detection on arbitrary annotated video collections.

    Args:
        root:             Dataset root directory.
        split:            Name of the split (matches splits/{split}.txt).
        class_names:      Ordered list of foreground class names to detect.
                          Any object whose class is not in this list is ignored.
        num_frames:       Frames per clip (paper default: 16).
        img_size:         Square resize target in pixels (paper default: 224 or 640).
        temporal_stride:  Frame stride within a clip (paper default: 4).
        clips_per_video:  How many clips to yield per video per epoch.
        use_depth:        If True, loads depth channel and returns 4-channel clips.
        train:            Randomise clip start (True) or use centre clip (False).
        transforms:       Optional (clip, target) → (clip, target) callable.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        class_names: Sequence[str] = ("face", "thorax"),
        num_frames: int = 16,
        img_size: int = 224,
        temporal_stride: int = 4,
        clips_per_video: int = 16,
        use_depth: bool = False,
        train: bool = True,
        transforms=None,
    ):
        self.root = Path(root)
        self.split = split
        self.class_names = list(class_names)
        self.class_to_id: Dict[str, int] = {n: i for i, n in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.num_frames = num_frames
        self.img_size = img_size
        self.temporal_stride = temporal_stride
        self.clips_per_video = clips_per_video
        self.use_depth = use_depth
        self.train = train
        self.transforms = transforms

        split_file = self.root / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Expected one video_id per line."
            )
        self.video_ids: List[str] = [
            ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()
        ]
        if not self.video_ids:
            raise ValueError(f"Split file is empty: {split_file}")

        self.ann_dir = self.root / "annotations"
        self.video_dir = self.root / "videos"

    # ------------------------------------------------------------------ #
    # Internal helpers

    def _load_annotation(self, video_id: str) -> dict:
        ann_path = self.ann_dir / f"{video_id}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation not found: {ann_path}")
        with open(ann_path) as f:
            return json.load(f)

    def _resolve_video_path(self, video_id: str, ann: dict) -> str:
        """Return the absolute path to the video file."""
        if "file" in ann:
            p = Path(ann["file"])
            return str(p if p.is_absolute() else self.root / p)
        return str(self.video_dir / f"{video_id}.mp4")

    def _resolve_depth_path(self, video_id: str, ann: dict) -> Optional[str]:
        if "depth_file" in ann:
            p = Path(ann["depth_file"])
            return str(p if p.is_absolute() else self.root / p)
        candidate = self.video_dir / f"{video_id}_depth.mp4"
        return str(candidate) if candidate.exists() else None

    def _sample_indices(self, total_frames: int) -> List[int]:
        span = self.temporal_stride * (self.num_frames - 1) + 1
        max_start = max(0, total_frames - span)
        if self.train:
            start = np.random.randint(0, max_start + 1)
        else:
            start = max_start // 2
        indices = [start + i * self.temporal_stride for i in range(self.num_frames)]
        return [min(i, total_frames - 1) for i in indices]

    def _build_target(self, ann: dict, frame_indices: List[int]) -> dict:
        """Aggregate per-frame annotations into one per-clip OBB target.

        Uses median (x,y,w,h) and circular mean angle across the selected frames.
        """
        img_h, img_w = ann["height"], ann["width"]
        num_classes = self.num_classes
        cls = np.zeros(num_classes, dtype=np.float32)
        bbox = np.zeros((num_classes, 4), dtype=np.float32)
        angle = np.zeros(num_classes, dtype=np.float32)

        by_class: Dict[int, list] = {c: [] for c in range(num_classes)}
        idx_set = set(frame_indices)
        for fr in ann.get("frames", []):
            if fr["index"] not in idx_set:
                continue
            for obj in fr.get("objects", []):
                cid = self.class_to_id.get(obj["class"], -1)
                if cid < 0:
                    continue
                theta_rad = math.radians(obj["theta_deg"]) % math.pi
                by_class[cid].append((obj["xc"], obj["yc"], obj["w"], obj["h"], theta_rad))

        for cid, items in by_class.items():
            if not items:
                continue
            arr = np.array(items, dtype=np.float32)
            med = np.median(arr, axis=0)
            cls[cid] = 1.0
            bbox[cid] = [med[0] / img_w, med[1] / img_h,
                         med[2] / img_w, med[3] / img_h]
            two_t = 2 * arr[:, 4]
            angle[cid] = 0.5 * (math.atan2(np.sin(two_t).mean(),
                                            np.cos(two_t).mean()) % (2 * math.pi))

        return {
            "cls":   torch.from_numpy(cls),
            "bbox":  torch.from_numpy(bbox),
            "angle": torch.from_numpy(angle),
        }

    # ------------------------------------------------------------------ #
    # Dataset interface

    def __len__(self) -> int:
        return len(self.video_ids) * self.clips_per_video

    def __getitem__(self, idx: int) -> dict:
        video_id = self.video_ids[idx % len(self.video_ids)]
        ann = self._load_annotation(video_id)

        img_h, img_w = ann["height"], ann["width"]
        total = max((fr["index"] for fr in ann.get("frames", [])), default=0) + 1
        frame_indices = self._sample_indices(total)

        video_path = self._resolve_video_path(video_id, ann)
        rgb = _load_video_frames(video_path, frame_indices, self.img_size)   # (T, 3, H, W)

        if self.use_depth:
            depth_path = self._resolve_depth_path(video_id, ann)
            if depth_path and Path(depth_path).exists():
                depth = _load_depth_frames(depth_path, frame_indices, self.img_size)
            else:
                depth = torch.zeros_like(rgb[:, :1])
            clip = torch.cat([rgb, depth], dim=1)                            # (T, 4, H, W)
        else:
            clip = rgb                                                         # (T, 3, H, W)

        clip = clip.permute(1, 0, 2, 3).contiguous()                          # (C, T, H, W)
        target = self._build_target(ann, frame_indices)

        if self.transforms is not None:
            clip, target = self.transforms(clip, target)

        return {
            "clip":          clip,
            "target":        target,
            "video_id":      video_id,
            "frame_indices": torch.tensor(frame_indices),
        }


def collate_fn(batch: List[dict]) -> dict:
    """Batch-collate clips and targets; preserve string video_ids separately."""
    clips = torch.stack([b["clip"] for b in batch])
    target = {
        "cls":   torch.stack([b["target"]["cls"]   for b in batch]),
        "bbox":  torch.stack([b["target"]["bbox"]  for b in batch]),
        "angle": torch.stack([b["target"]["angle"] for b in batch]),
    }
    return {
        "clip":          clips,
        "target":        target,
        "video_ids":     [b["video_id"] for b in batch],
        "frame_indices": torch.stack([b["frame_indices"] for b in batch]),
    }
