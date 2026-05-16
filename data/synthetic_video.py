"""Synthetic video clip generation for self-supervised pretraining.

Implements the augmentation strategy described in Sec. 4.1.2: take a single
hospital/NICU/PICU image and synthesize a short video clip by progressively
applying small spatial and photometric perturbations across frames.

Outputs:
    output_dir/
        clip_000000.mp4
        clip_000001.mp4
        ...

Run as:
    python -m data.synthetic_video --input_dir <images> --output_dir <clips> \
        --num_frames 16 --target_clips 15000
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm


def random_perturbation_schedule(num_frames: int) -> dict:
    """Sample a smooth per-frame perturbation schedule."""
    return {
        "rotation_deg": np.cumsum(np.random.normal(0, 0.5, num_frames)),
        "dx_frac":      np.cumsum(np.random.normal(0, 0.005, num_frames)),
        "dy_frac":      np.cumsum(np.random.normal(0, 0.005, num_frames)),
        "scale":        1.0 + np.cumsum(np.random.normal(0, 0.002, num_frames)),
        "brightness":   np.cumsum(np.random.normal(0, 1.5, num_frames)),
        "contrast":     1.0 + np.cumsum(np.random.normal(0, 0.01, num_frames)),
        "noise_std":    np.abs(np.random.normal(2.0, 1.0, num_frames)),
    }


def apply_frame(img: np.ndarray, sch: dict, t: int) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(sch["rotation_deg"][t]), float(sch["scale"][t]))
    M[0, 2] += float(sch["dx_frac"][t]) * w
    M[1, 2] += float(sch["dy_frac"][t]) * h
    frame = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    frame = frame.astype(np.float32)
    frame = frame * float(sch["contrast"][t]) + float(sch["brightness"][t])
    if sch["noise_std"][t] > 0:
        frame += np.random.normal(0, float(sch["noise_std"][t]), frame.shape)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def synthesize_clip(image_path: str, num_frames: int = 16,
                    target_size: int = 256) -> List[np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        return []
    img = cv2.resize(img, (target_size, target_size))
    sch = random_perturbation_schedule(num_frames)
    return [apply_frame(img, sch, t) for t in range(num_frames)]


def write_video(frames: List[np.ndarray], path: str, fps: int = 15) -> None:
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory of source images.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--target_size", type=int, default=256)
    ap.add_argument("--target_clips", type=int, default=15000)
    ap.add_argument("--clips_per_image", type=int, default=3)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for ext in ("jpg", "jpeg", "png", "bmp"):
        images.extend(sorted(Path(args.input_dir).rglob(f"*.{ext}")))
    if not images:
        raise RuntimeError(f"No images found under {args.input_dir}")

    print(f"Found {len(images)} images. Generating ~{args.target_clips} clips.")
    n_written = 0
    pbar = tqdm(total=args.target_clips)
    while n_written < args.target_clips:
        img_path = str(random.choice(images))
        for _ in range(args.clips_per_image):
            if n_written >= args.target_clips:
                break
            frames = synthesize_clip(img_path, args.num_frames, args.target_size)
            if not frames:
                continue
            write_video(frames, str(out_dir / f"clip_{n_written:06d}.mp4"), fps=args.fps)
            n_written += 1
            pbar.update(1)
    pbar.close()
    print(f"Wrote {n_written} clips to {out_dir}")


if __name__ == "__main__":
    main()
