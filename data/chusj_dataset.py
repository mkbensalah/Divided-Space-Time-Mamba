"""CHU Sainte-Justine PICU video dataset.

Each sample is a fixed-length clip of T frames sampled from a patient video,
with optional aligned depth. Annotations are oriented bounding boxes per frame
(face and thoracoabdominal), aggregated to a single per-clip OBB per class
(the median over visible frames, since the patient is roughly stationary
within a clip).

Expected on-disk layout (see README):
    root/
        videos/{patient_id}.mp4
        videos/{patient_id}_depth.mp4    (optional, for RGB-D)
        annotations/{patient_id}.json
        splits/{train,val,test}_patients.txt

Annotation JSON schema (per patient):
    {
        "frames": [
            {
                "index": int,
                "objects": [
                    {"class": "face"|"thorax",
                     "xc": float, "yc": float,
                     "w": float, "h": float,
                     "theta_deg": float}
                ]
            },
            ...
        ],
        "fps": int,
        "height": int,
        "width": int
    }
Coordinates are pixel-space at native resolution.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import decord
    decord.bridge.set_bridge("torch")
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

import cv2

CLASS_NAMES = ["face", "thorax"]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def _load_video_frames(path: str, indices: List[int], target_size: int) -> torch.Tensor:
    """Load specified frame indices from a video file, resized to square target_size."""
    if HAS_DECORD:
        vr = decord.VideoReader(path)
        frames = vr.get_batch(indices)                     # (T, H, W, C) uint8 tensor
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
    else:
        cap = cv2.VideoCapture(path)
        idx_set = set(indices)
        max_idx = max(indices)
        frames_dict = {}
        i = 0
        while i <= max_idx:
            ok, frame = cap.read()
            if not ok:
                break
            if i in idx_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_dict[i] = frame
            i += 1
        cap.release()
        arr = np.stack([frames_dict[i] for i in indices], axis=0)
        frames = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0

    # Resize
    frames = torch.nn.functional.interpolate(
        frames, size=(target_size, target_size), mode="bilinear", align_corners=False
    )
    return frames  # (T, C, H, W)


def _load_depth_frames(path: str, indices: List[int], target_size: int) -> torch.Tensor:
    """Load depth frames (single channel)."""
    if HAS_DECORD:
        vr = decord.VideoReader(path)
        frames = vr.get_batch(indices).float()             # (T, H, W, C)
        # Depth saved as grayscale; take channel 0.
        frames = frames[..., :1].permute(0, 3, 1, 2) / 255.0
    else:
        cap = cv2.VideoCapture(path)
        idx_set = set(indices)
        max_idx = max(indices)
        frames_dict = {}
        i = 0
        while i <= max_idx:
            ok, frame = cap.read()
            if not ok:
                break
            if i in idx_set:
                # Convert to grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames_dict[i] = frame
            i += 1
        cap.release()
        arr = np.stack([frames_dict[i] for i in indices], axis=0)[..., None]
        frames = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0

    frames = torch.nn.functional.interpolate(
        frames, size=(target_size, target_size), mode="bilinear", align_corners=False
    )
    return frames


def _aggregate_clip_obb(annotations: List[dict], frame_indices: List[int],
                        img_h: int, img_w: int) -> dict:
    """Compute per-clip OBB target from per-frame annotations.

    Returns:
        cls:   (num_classes,) float
        bbox:  (num_classes, 4) normalized (xc, yc, w, h) in [0, 1]
        angle: (num_classes,)   radians in [0, pi)
    """
    num_classes = len(CLASS_NAMES)
    cls = np.zeros(num_classes, dtype=np.float32)
    bbox = np.zeros((num_classes, 4), dtype=np.float32)
    angle = np.zeros(num_classes, dtype=np.float32)

    # Gather per-class lists across selected frames.
    by_class = {c: [] for c in range(num_classes)}
    idx_set = set(frame_indices)
    for fr in annotations:
        if fr["index"] not in idx_set:
            continue
        for obj in fr.get("objects", []):
            cid = CLASS_TO_ID.get(obj["class"], -1)
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
        bbox[cid] = np.array([med[0] / img_w, med[1] / img_h,
                              med[2] / img_w, med[3] / img_h])
        # Circular median for angle: use mean of unit vectors at 2*theta.
        two_t = 2 * arr[:, 4]
        mean_cos = np.cos(two_t).mean()
        mean_sin = np.sin(two_t).mean()
        angle[cid] = 0.5 * (math.atan2(mean_sin, mean_cos) % (2 * math.pi))

    return {"cls": torch.from_numpy(cls),
            "bbox": torch.from_numpy(bbox),
            "angle": torch.from_numpy(angle)}


class CHUSJVideoDataset(Dataset):
    """PICU video dataset with optional depth modality.

    Args:
        root:           Dataset root directory.
        split:          'train', 'val', or 'test' (uses splits/*.txt).
        num_frames:     Frames per clip (paper: 16).
        img_size:       Spatial resolution (paper: 224 or 640).
        temporal_stride: Stride between sampled frames within a clip.
        clips_per_patient: Number of clips to sample per patient at __len__.
                          Paper uses uniform sampling across the 30-s recording.
        use_depth:      If True, returns 4-channel RGB-D clips.
        train:          If True, randomized clip sampling; else deterministic.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: int = 16,
        img_size: int = 224,
        temporal_stride: int = 4,
        clips_per_patient: int = 16,
        use_depth: bool = False,
        train: bool = True,
        transforms=None,
    ):
        self.root = Path(root)
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.temporal_stride = temporal_stride
        self.clips_per_patient = clips_per_patient
        self.use_depth = use_depth
        self.train = train
        self.transforms = transforms

        split_file = self.root / "splits" / f"{split}_patients.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file missing: {split_file}")
        self.patients = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]

        self.video_dir = self.root / "videos"
        self.ann_dir = self.root / "annotations"

    def __len__(self) -> int:
        return len(self.patients) * self.clips_per_patient

    def _sample_indices(self, total_frames: int) -> List[int]:
        span = self.temporal_stride * (self.num_frames - 1) + 1
        max_start = max(0, total_frames - span)
        if self.train:
            start = np.random.randint(0, max_start + 1)
        else:
            # Deterministic uniform sampling for eval.
            start = max_start // 2
        return [start + i * self.temporal_stride for i in range(self.num_frames)]

    def __getitem__(self, idx: int) -> dict:
        patient_id = self.patients[idx % len(self.patients)]
        ann_path = self.ann_dir / f"{patient_id}.json"
        video_path = self.video_dir / f"{patient_id}.mp4"

        with open(ann_path, "r") as f:
            ann = json.load(f)
        img_h, img_w = ann["height"], ann["width"]

        # Determine total frames from annotations (or video stream).
        total = max(fr["index"] for fr in ann["frames"]) + 1
        frame_indices = self._sample_indices(total)

        rgb = _load_video_frames(str(video_path), frame_indices, self.img_size)  # (T, 3, H, W)

        if self.use_depth:
            depth_path = self.video_dir / f"{patient_id}_depth.mp4"
            if depth_path.exists():
                depth = _load_depth_frames(str(depth_path), frame_indices, self.img_size)
                clip = torch.cat([rgb, depth], dim=1)         # (T, 4, H, W)
            else:
                # Pad with zeros if depth is missing.
                depth = torch.zeros_like(rgb[:, :1])
                clip = torch.cat([rgb, depth], dim=1)
        else:
            clip = rgb

        # (T, C, H, W) → (C, T, H, W)
        clip = clip.permute(1, 0, 2, 3).contiguous()

        target = _aggregate_clip_obb(ann["frames"], frame_indices, img_h, img_w)

        if self.transforms is not None:
            clip, target = self.transforms(clip, target)

        return {"clip": clip, "target": target, "patient_id": patient_id,
                "frame_indices": torch.tensor(frame_indices)}


def collate_fn(batch: List[dict]) -> dict:
    clips = torch.stack([b["clip"] for b in batch])
    target = {
        "cls":   torch.stack([b["target"]["cls"] for b in batch]),
        "bbox":  torch.stack([b["target"]["bbox"] for b in batch]),
        "angle": torch.stack([b["target"]["angle"] for b in batch]),
    }
    patient_ids = [b["patient_id"] for b in batch]
    frame_indices = torch.stack([b["frame_indices"] for b in batch])
    return {"clip": clips, "target": target,
            "patient_ids": patient_ids, "frame_indices": frame_indices}
