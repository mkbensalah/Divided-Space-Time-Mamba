"""Evaluate a fine-tuned DST-Mamba checkpoint on any VideoOBBDataset split.

Reports mAP at multiple IoU thresholds, mAP50-95, rotated IoU, angle accuracy,
and temporal IoU.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import VideoOBBDataset, collate_fn, default_eval_transforms
from utils import csl_decode, rotated_iou_matrix, angle_accuracy, mean_average_precision, temporal_iou
from scripts.finetune import Detector


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--score_threshold", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Detector(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    class_names = cfg["data"].get("class_names", ["face", "thorax"])
    clips_per_video = cfg["data"].get("clips_per_video", 16)
    ds = VideoOBBDataset(
        root=args.data_root, split=args.split, train=False,
        class_names=class_names,
        num_frames=cfg["data"]["num_frames"], img_size=cfg["data"]["img_size"],
        temporal_stride=cfg["data"]["temporal_stride"],
        clips_per_video=clips_per_video,
        use_depth=cfg["data"]["use_depth"],
        transforms=default_eval_transforms(),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn)

    preds_all, gts_all = [], []
    pred_angles, gt_angles = [], []
    pred_obbs_pos, gt_obbs_pos = [], []

    # Per-patient, per-class sequences of (frame_start, obb) for temporal IoU.
    # Temporal IoU measures how stable detections are across clips of the same patient.
    per_patient_pred = defaultdict(lambda: defaultdict(list))

    num_classes = len(class_names)

    for batch in loader:
        clip = batch["clip"].to(device, non_blocking=True)
        out = model(clip)
        prob = torch.sigmoid(out["cls"]).cpu()
        bbox = out["bbox"].cpu()
        theta = csl_decode(out["angle"]).cpu()

        B = prob.shape[0]
        for b in range(B):
            pid = batch["video_ids"][b]
            frame_start = batch["frame_indices"][b][0].item()
            present = prob[b] > args.score_threshold
            keep_b, keep_s, keep_l = [], [], []
            for c in range(num_classes):
                if not present[c]:
                    continue
                x, y, w, h = bbox[b, c]
                obb = torch.tensor([x, y, w, h, theta[b, c]])
                keep_b.append(obb)
                keep_s.append(prob[b, c])
                keep_l.append(c)
                # Accumulate for temporal IoU: sort by frame_start later.
                per_patient_pred[pid][c].append((frame_start, obb))
            preds_all.append({
                "boxes":  torch.stack(keep_b) if keep_b else torch.zeros(0, 5),
                "scores": torch.stack(keep_s) if keep_s else torch.zeros(0),
                "labels": torch.tensor(keep_l) if keep_l else torch.zeros(0, dtype=torch.long),
            })

            g_cls = batch["target"]["cls"][b]
            g_bbox = batch["target"]["bbox"][b]
            g_angle = batch["target"]["angle"][b]
            g_b, g_l = [], []
            for c in range(num_classes):
                if g_cls[c] > 0.5:
                    g_b.append(torch.cat([g_bbox[c], g_angle[c:c + 1]]))
                    g_l.append(c)
                    if present[c]:
                        pred_obbs_pos.append(torch.tensor(
                            [bbox[b, c, 0], bbox[b, c, 1], bbox[b, c, 2], bbox[b, c, 3], theta[b, c]]
                        ))
                        gt_obbs_pos.append(torch.cat([g_bbox[c], g_angle[c:c + 1]]))
                        pred_angles.append(theta[b, c])
                        gt_angles.append(g_angle[c])
            gts_all.append({
                "boxes":  torch.stack(g_b) if g_b else torch.zeros(0, 5),
                "labels": torch.tensor(g_l) if g_l else torch.zeros(0, dtype=torch.long),
            })

    thrs = tuple(cfg["eval"]["iou_thresholds"])
    metrics = mean_average_precision(preds_all, gts_all, num_classes, iou_thrs=thrs)

    if pred_obbs_pos:
        p_obb = torch.stack(pred_obbs_pos)
        g_obb = torch.stack(gt_obbs_pos)
        from losses.rotated_iou import rotated_iou_shapely
        riou = rotated_iou_shapely(p_obb, g_obb).mean().item()
        ang_acc = angle_accuracy(torch.stack(pred_angles), torch.stack(gt_angles))
    else:
        riou, ang_acc = 0.0, 0.0

    # Temporal IoU: mean consecutive-clip rIoU per patient per class.
    # Clips are sorted by their starting frame index to reflect temporal order.
    tiou_vals = []
    for pid, class_clips in per_patient_pred.items():
        for c, clip_list in class_clips.items():
            clip_list.sort(key=lambda item: item[0])          # sort by frame_start
            obbs = [item[1] for item in clip_list]
            if len(obbs) >= 2:
                tiou_vals.append(temporal_iou(obbs))
    t_iou = float(np.mean(tiou_vals)) if tiou_vals else 0.0

    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split} | {len(ds)} clips")
    print("-" * 60)
    for k, v in metrics.items():
        print(f"  {k:>14s}: {v:.4f}")
    print(f"  {'rIoU':>14s}: {riou:.4f}")
    print(f"  {'Angle acc':>14s}: {ang_acc:.4f}")
    print(f"  {'Temporal IoU':>14s}: {t_iou:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
