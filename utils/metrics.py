"""Evaluation metrics for OBB detection.

Provides:
    - rotated_iou_matrix:   pairwise rIoU between sets of OBBs.
    - mean_average_precision: mAP over IoU thresholds for OBB detection.
    - temporal_iou:           consecutive-frame OBB stability (Sec. 4.3.2).
    - angle_accuracy:         exact-bin angle accuracy.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

from losses.rotated_iou import rotated_iou_shapely, obb_to_corners


def rotated_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """All-pairs rotated IoU between two sets of OBBs.

    Args:
        boxes1: (N, 5), boxes2: (M, 5), each (xc, yc, w, h, theta_rad).
    Returns:
        (N, M) IoU matrix.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(boxes1.shape[0], boxes2.shape[0])
    N, M = boxes1.shape[0], boxes2.shape[0]
    b1 = boxes1.unsqueeze(1).expand(N, M, 5).reshape(-1, 5)
    b2 = boxes2.unsqueeze(0).expand(N, M, 5).reshape(-1, 5)
    iou = rotated_iou_shapely(b1, b2)
    return iou.view(N, M)


def average_precision_at_iou(
    preds: Sequence[dict],
    gts: Sequence[dict],
    class_id: int,
    iou_thr: float,
) -> float:
    """Average precision for one class at one IoU threshold.

    Args:
        preds: list of per-clip dicts with keys 'boxes' (N, 5), 'scores' (N,), 'labels' (N,).
        gts:   list of per-clip dicts with keys 'boxes' (M, 5), 'labels' (M,).
        class_id: class to evaluate.
        iou_thr: IoU threshold (e.g. 0.5).
    Returns:
        AP in [0, 1].
    """
    # Collect predictions
    all_scores, all_tp, all_fp = [], [], []
    num_gt = 0
    matched_gt = []

    for pred, gt in zip(preds, gts):
        p_mask = pred["labels"] == class_id
        g_mask = gt["labels"] == class_id
        p_boxes = pred["boxes"][p_mask]
        p_scores = pred["scores"][p_mask]
        g_boxes = gt["boxes"][g_mask]
        num_gt += g_boxes.shape[0]

        if p_boxes.shape[0] == 0:
            continue
        if g_boxes.shape[0] == 0:
            all_scores.append(p_scores)
            all_tp.append(torch.zeros_like(p_scores))
            all_fp.append(torch.ones_like(p_scores))
            continue

        # Sort predictions by score descending
        order = torch.argsort(p_scores, descending=True)
        p_boxes, p_scores = p_boxes[order], p_scores[order]
        iou = rotated_iou_matrix(p_boxes, g_boxes)                 # (Np, Mg)

        gt_used = torch.zeros(g_boxes.shape[0], dtype=torch.bool)
        tp = torch.zeros(p_boxes.shape[0])
        fp = torch.zeros(p_boxes.shape[0])
        for i in range(p_boxes.shape[0]):
            best_j = torch.argmax(iou[i])
            if iou[i, best_j] >= iou_thr and not gt_used[best_j]:
                tp[i] = 1.0
                gt_used[best_j] = True
            else:
                fp[i] = 1.0
        all_scores.append(p_scores)
        all_tp.append(tp)
        all_fp.append(fp)

    if num_gt == 0 or len(all_scores) == 0:
        return 0.0

    scores = torch.cat(all_scores)
    tp = torch.cat(all_tp)
    fp = torch.cat(all_fp)
    order = torch.argsort(scores, descending=True)
    tp, fp = tp[order], fp[order]
    cum_tp = torch.cumsum(tp, dim=0)
    cum_fp = torch.cumsum(fp, dim=0)
    recall = cum_tp / max(num_gt, 1)
    precision = cum_tp / (cum_tp + cum_fp).clamp(min=1e-8)

    # 101-point interpolation (COCO style)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recall >= t
        p = precision[mask].max().item() if mask.any() else 0.0
        ap += p / 101
    return float(ap)


def mean_average_precision(preds, gts, num_classes: int, iou_thrs=(0.5,)) -> dict:
    """mAP averaged over classes and IoU thresholds.

    Returns dict mapping each threshold (and 'mAP@50-95' if multiple) to its value.
    """
    out = {}
    per_thr = []
    for thr in iou_thrs:
        ap_per_class = [average_precision_at_iou(preds, gts, c, thr)
                        for c in range(num_classes)]
        out[f"mAP@{thr:.2f}"] = float(np.mean(ap_per_class))
        per_thr.append(out[f"mAP@{thr:.2f}"])
    if len(iou_thrs) > 1:
        out["mAP50-95"] = float(np.mean(per_thr))
    return out


def temporal_iou(per_frame_boxes: List[torch.Tensor]) -> float:
    """Temporal IoU = mean IoU between detections in consecutive frames.

    Args:
        per_frame_boxes: list of (5,) OBBs (one detection per frame).
    Returns:
        scalar in [0, 1].
    """
    if len(per_frame_boxes) < 2:
        return 0.0
    ious = []
    for t in range(len(per_frame_boxes) - 1):
        b1 = per_frame_boxes[t].unsqueeze(0)
        b2 = per_frame_boxes[t + 1].unsqueeze(0)
        ious.append(rotated_iou_shapely(b1, b2).item())
    return float(np.mean(ious))


def angle_accuracy(pred_angles_rad: torch.Tensor, gt_angles_rad: torch.Tensor,
                   num_bins: int = 180) -> float:
    """Fraction of predictions whose CSL bin matches the GT bin exactly."""
    import math
    pred_bin = ((pred_angles_rad % math.pi) / math.pi * num_bins).long().clamp(max=num_bins - 1)
    gt_bin = ((gt_angles_rad % math.pi) / math.pi * num_bins).long().clamp(max=num_bins - 1)
    return (pred_bin == gt_bin).float().mean().item()
