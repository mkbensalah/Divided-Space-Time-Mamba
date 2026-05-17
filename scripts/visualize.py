"""Render qualitative detection results.

For each requested video, samples a clip, runs DST-Mamba, and overlays both
ground-truth and predicted oriented bounding boxes on representative frames.

    GT solid:    first class = green,  second class = blue
    Pred dashed: first class = cyan,   second class = yellow

Outputs PNGs per video plus a combined PDF figure.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import VideoOBBDataset, collate_fn, default_eval_transforms
from utils import csl_decode
from utils.obb import obb_to_corners_np, denormalize_obb
from scripts.finetune import Detector

GT_COLORS = [(0, 255, 0), (255, 0, 0), (0, 165, 255), (128, 0, 128)]      # BGR
PRED_COLORS = [(255, 255, 0), (0, 255, 255), (0, 128, 255), (255, 0, 255)] # BGR


def draw_obb(img: np.ndarray, obb_pix: np.ndarray, color, dashed: bool = False, thickness: int = 2):
    pts = obb_pix.astype(np.int32)
    for i in range(4):
        p1, p2 = tuple(pts[i]), tuple(pts[(i + 1) % 4])
        if dashed:
            d = np.array(p2) - np.array(p1)
            length = np.linalg.norm(d)
            if length == 0:
                continue
            n = int(length / 8)
            for k in range(0, n, 2):
                a = (np.array(p1) + d * (k / n)).astype(np.int32)
                b = (np.array(p1) + d * ((k + 1) / n)).astype(np.int32)
                cv2.line(img, tuple(a), tuple(b), color, thickness)
        else:
            cv2.line(img, p1, p2, color, thickness)


def denorm_clip_for_display(clip: torch.Tensor) -> np.ndarray:
    """(C, T, H, W) normalized → (T, H, W, 3) uint8 BGR."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    rgb = clip[:3] * std + mean
    rgb = rgb.clamp(0, 1).permute(1, 2, 3, 0).numpy()
    return (rgb[..., ::-1] * 255).astype(np.uint8)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--videos", nargs="+", required=True,
                   help="Video IDs to visualize (must match split file entries).")
    p.add_argument("--split", default="test")
    p.add_argument("--output_dir", default="figures/qualitative")
    p.add_argument("--num_frames_to_save", type=int, default=4)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Detector(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])

    class_names = cfg["data"].get("class_names", ["face", "thorax"])

    # Write a temporary split file containing only the requested videos.
    tmp_split = Path(args.data_root) / "splits" / "_visualize_tmp.txt"
    tmp_split.write_text("\n".join(args.videos) + "\n")

    ds = VideoOBBDataset(
        root=args.data_root, split="_visualize_tmp", train=False,
        class_names=class_names,
        num_frames=cfg["data"]["num_frames"], img_size=cfg["data"]["img_size"],
        temporal_stride=cfg["data"]["temporal_stride"],
        clips_per_video=1,
        use_depth=cfg["data"]["use_depth"],
        transforms=default_eval_transforms(),
    )
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)

    panels = []
    for batch in loader:
        vid = batch["video_ids"][0]
        clip = batch["clip"].to(device)
        out = model(clip)
        prob = torch.sigmoid(out["cls"])[0].cpu()
        bbox = out["bbox"][0].cpu()
        theta = csl_decode(out["angle"])[0].cpu()

        g_cls = batch["target"]["cls"][0]
        g_bbox = batch["target"]["bbox"][0]
        g_angle = batch["target"]["angle"][0]

        frames = denorm_clip_for_display(batch["clip"][0].cpu())
        H, W = frames.shape[1:3]
        step = max(1, frames.shape[0] // args.num_frames_to_save)

        vid_out = out_dir / vid
        vid_out.mkdir(exist_ok=True)
        chosen_frames = []
        for ti in range(0, frames.shape[0], step):
            frame = frames[ti].copy()
            for cid, cname in enumerate(class_names):
                gt_color = GT_COLORS[cid % len(GT_COLORS)]
                pred_color = PRED_COLORS[cid % len(PRED_COLORS)]
                if g_cls[cid] > 0.5:
                    gobb = torch.cat([g_bbox[cid], g_angle[cid:cid + 1]]).unsqueeze(0)
                    gobb = denormalize_obb(gobb, H, W).numpy()[0]
                    draw_obb(frame, obb_to_corners_np(np.array([gobb]))[0], gt_color, dashed=False)
                if prob[cid] > 0.3:
                    pobb = torch.tensor([bbox[cid, 0], bbox[cid, 1],
                                         bbox[cid, 2], bbox[cid, 3], theta[cid]]).unsqueeze(0)
                    pobb = denormalize_obb(pobb, H, W).numpy()[0]
                    draw_obb(frame, obb_to_corners_np(np.array([pobb]))[0], pred_color, dashed=True)
                    cv2.putText(frame, f"{cname}:{prob[cid]:.2f}",
                                (10, 30 + cid * 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, pred_color, 2)

            cv2.imwrite(str(vid_out / f"frame_{ti:04d}.png"), frame)
            chosen_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(chosen_frames) >= args.num_frames_to_save:
                break
        panels.append((vid, chosen_frames))

    n_vid = len(panels)
    n_col = args.num_frames_to_save
    fig, axes = plt.subplots(n_vid, n_col, figsize=(3.5 * n_col, 3.5 * n_vid))
    if n_vid == 1:
        axes = [axes]
    for i, (vid, frames) in enumerate(panels):
        row = axes[i] if n_col > 1 else [axes[i]]
        for j, frame in enumerate(frames):
            row[j].imshow(frame)
            row[j].axis("off")
            if j == 0:
                row[j].set_ylabel(vid, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "results.pdf", dpi=200, bbox_inches="tight")
    plt.close()

    tmp_split.unlink(missing_ok=True)
    print(f"Saved visualizations to {out_dir}")


if __name__ == "__main__":
    main()
