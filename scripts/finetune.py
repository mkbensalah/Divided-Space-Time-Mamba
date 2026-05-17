"""Fine-tune DST-Mamba for oriented bounding box detection.

Works with any dataset that follows the VideoOBBDataset layout:
    root/
        annotations/{video_id}.json
        splits/{train,val,test}.txt
        videos/{video_id}.mp4

Usage:
    python scripts/finetune.py \
        --config configs/finetune_detection.yaml \
        --pretrained runs/pretrain/checkpoint-last.pth \
        --data_root data/my_dataset \
        --output_dir runs/finetune
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import (
    VideoOBBDataset, collate_fn,
    default_train_transforms, default_eval_transforms,
)
from losses import DetectionLoss
from models import DSTMamba, OBBDetectionHead
from utils import mean_average_precision, csl_decode


# --------------------------------------------------------------------- #

class Detector(nn.Module):
    """DST-Mamba backbone + OBB detection head."""
    def __init__(self, cfg):
        super().__init__()
        self.backbone = DSTMamba(**cfg["model"], cls_token=False)
        # num_classes is derived from class_names so config stays consistent.
        num_classes = len(cfg["data"].get("class_names", ["face", "thorax"]))
        self.head = OBBDetectionHead(
            embed_dim=cfg["model"]["embed_dim"],
            num_classes=num_classes,
            num_angle_bins=cfg["head"]["num_angle_bins"],
            hidden_dim=cfg["head"]["hidden_dim"],
        )

    def forward(self, x):
        tokens = self.backbone(x)
        return self.head(tokens)


def load_pretrained_encoder(model: Detector, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Try the "encoder" key first (saved by pretraining), then fall back to "model".
    if "encoder" in ckpt:
        state = ckpt["encoder"]
    else:
        # Strip 'encoder.' prefix from MAE wrapper checkpoint.
        state = {k.replace("encoder.", "", 1): v for k, v in ckpt["model"].items()
                 if k.startswith("encoder.")}
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    print(f"[finetune] loaded pretrained backbone. missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print(f"  unexpected (first 5): {unexpected[:5]}")


def cosine_with_warmup(step, total, warmup, base_lr, min_lr=1e-6):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def build_optimizer(model: Detector, cfg: dict):
    """RAdam with layer-wise LR decay (0.85 per layer) on the backbone.

    Schedule (from deepest to shallowest):
        patch_embed / pos_embed / norm:  base_lr * decay^num_layers
        blocks[i]:                       base_lr * decay^(num_layers - i)
        detection head:                  base_lr  (no decay)

    The '_base_lr' key stored in each param group is used by the training loop
    to apply the cosine schedule proportionally across all groups.
    """
    base_lr = cfg["optim"]["lr"]
    wd = cfg["optim"]["weight_decay"]
    decay = cfg["optim"].get("layer_wise_lr_decay", 1.0)
    num_layers = len(model.backbone.blocks)

    try:
        from torch.optim import RAdam
    except ImportError:
        from timm.optim import RAdam

    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "backbone.blocks." in name:
            layer_idx = int(name.split("backbone.blocks.")[1].split(".")[0])
            # Block 0 (earliest) gets the most decay; block num_layers-1 gets decay^1.
            group_lr = base_lr * (decay ** (num_layers - layer_idx))
        elif "backbone." in name:
            # patch_embed, positional embeddings, final norm: treat as layer 0.
            group_lr = base_lr * (decay ** num_layers)
        else:
            # Detection head: full learning rate, no decay.
            group_lr = base_lr

        # No weight decay on 1-D params (norms, biases).
        this_wd = 0.0 if (param.ndim == 1 or "norm" in name or "bias" in name) else wd
        param_groups.append({"params": [param], "lr": group_lr,
                             "weight_decay": this_wd, "_base_lr": group_lr})

    return RAdam(param_groups)


# --------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model, loader, device, score_threshold=0.3, iou_thresholds=(0.5,)):
    model.eval()
    preds_all, gts_all = [], []
    num_classes = model.head.num_classes

    for batch in loader:
        clip = batch["clip"].to(device, non_blocking=True)
        out = model(clip)
        prob = torch.sigmoid(out["cls"]).cpu()                       # (B, C)
        bbox = out["bbox"].cpu()                                     # (B, C, 4)
        theta = csl_decode(out["angle"]).cpu()                       # (B, C)

        B = prob.shape[0]
        for b in range(B):
            present = prob[b] > score_threshold
            keep_boxes = []
            keep_scores = []
            keep_labels = []
            for c in range(num_classes):
                if not present[c]:
                    continue
                x, y, w, h = bbox[b, c]
                keep_boxes.append(torch.tensor([x, y, w, h, theta[b, c]]))
                keep_scores.append(prob[b, c])
                keep_labels.append(c)
            if keep_boxes:
                preds_all.append({
                    "boxes":  torch.stack(keep_boxes),
                    "scores": torch.stack(keep_scores),
                    "labels": torch.tensor(keep_labels),
                })
            else:
                preds_all.append({
                    "boxes":  torch.zeros(0, 5),
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.long),
                })

            g_cls = batch["target"]["cls"][b]
            g_bbox = batch["target"]["bbox"][b]
            g_angle = batch["target"]["angle"][b]
            g_boxes, g_labels = [], []
            for c in range(num_classes):
                if g_cls[c] > 0.5:
                    g_boxes.append(torch.cat([g_bbox[c], g_angle[c:c + 1]]))
                    g_labels.append(c)
            gts_all.append({
                "boxes":  torch.stack(g_boxes) if g_boxes else torch.zeros(0, 5),
                "labels": torch.tensor(g_labels) if g_labels else torch.zeros(0, dtype=torch.long),
            })

    return mean_average_precision(preds_all, gts_all, num_classes, iou_thrs=iou_thresholds)


# --------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pretrained", default=None)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out / "tb")

    torch.manual_seed(cfg["train"]["seed"])
    np.random.seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_names = cfg["data"].get("class_names", ["face", "thorax"])
    clips_per_video = cfg["data"].get("clips_per_video", 16)

    # Datasets — VideoOBBDataset works with any annotated video collection.
    train_ds = VideoOBBDataset(
        root=args.data_root, split="train", train=True,
        class_names=class_names,
        num_frames=cfg["data"]["num_frames"], img_size=cfg["data"]["img_size"],
        temporal_stride=cfg["data"]["temporal_stride"],
        clips_per_video=clips_per_video,
        use_depth=cfg["data"]["use_depth"],
        transforms=default_train_transforms(),
    )
    val_split = "val" if (Path(args.data_root) / "splits" / "val.txt").exists() else "test"
    val_ds = VideoOBBDataset(
        root=args.data_root, split=val_split, train=False,
        class_names=class_names,
        num_frames=cfg["data"]["num_frames"], img_size=cfg["data"]["img_size"],
        temporal_stride=cfg["data"]["temporal_stride"],
        clips_per_video=clips_per_video,
        use_depth=cfg["data"]["use_depth"],
        transforms=default_eval_transforms(),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["train"]["num_workers"], pin_memory=True,
                              drop_last=True, collate_fn=collate_fn,
                              persistent_workers=cfg["train"]["num_workers"] > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=cfg["train"]["num_workers"], pin_memory=True,
                            collate_fn=collate_fn)

    # Model
    model = Detector(cfg).to(device)
    if args.pretrained:
        load_pretrained_encoder(model, args.pretrained)

    # Loss
    criterion = DetectionLoss(
        num_classes=len(class_names),
        num_angle_bins=cfg["head"]["num_angle_bins"],
        alpha=cfg["loss"]["alpha"],
        beta=cfg["loss"]["beta"],
        csl_radius=cfg["loss"]["csl_radius"],
    )

    # Optimizer: RAdam with layer-wise LR decay (paper: 0.85 per backbone layer).
    optim = build_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"])

    steps_per_epoch = len(train_loader) // cfg["train"]["accum_steps"]
    total_steps = steps_per_epoch * cfg["optim"]["epochs"]
    warmup_steps = steps_per_epoch * cfg["optim"]["warmup_epochs"]

    best = -1.0
    global_step = 0
    save_key = cfg["train"]["save_best_metric"]

    for epoch in range(cfg["optim"]["epochs"]):
        model.train()
        t0 = time.time()
        running = {"total": 0, "cls": 0, "bbox": 0, "angle": 0, "iou": 0, "n": 0}
        for it, batch in enumerate(train_loader):
            clip = batch["clip"].to(device, non_blocking=True)
            tgt = {k: v.to(device, non_blocking=True) for k, v in batch["target"].items()}

            # Compute the schedule factor relative to base_lr so each param group
            # stays at its layer-decayed ratio throughout warmup and cosine decay.
            lr = cosine_with_warmup(global_step, total_steps, warmup_steps, cfg["optim"]["lr"])
            factor = lr / cfg["optim"]["lr"]
            for g in optim.param_groups:
                g["lr"] = g["_base_lr"] * factor

            with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"]):
                preds = model(clip)
                loss_dict = criterion(preds, tgt)
                loss = loss_dict["total"] / cfg["train"]["accum_steps"]
            scaler.scale(loss).backward()

            if (it + 1) % cfg["train"]["accum_steps"] == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                global_step += 1

            for k in ("total", "cls", "bbox", "angle", "iou"):
                running[k] += float(loss_dict[k])
            running["n"] += 1

            if (it + 1) % cfg["train"]["log_interval"] == 0:
                n = running["n"]
                print(f"[ep {epoch:3d}][it {it:4d}/{len(train_loader)}] "
                      f"L={running['total']/n:.3f} "
                      f"cls={running['cls']/n:.3f} bbox={running['bbox']/n:.3f} "
                      f"ang={running['angle']/n:.3f} iou={running['iou']/n:.3f} "
                      f"lr={lr:.2e} ({(time.time()-t0)/(it+1)*1000:.0f}ms/it)")
                writer.add_scalar("train/loss", running['total']/n, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                running = {k: 0 for k in running}

        # Validation
        metrics = evaluate(model, val_loader, device,
                           score_threshold=cfg["eval"]["score_threshold"],
                           iou_thresholds=tuple(cfg["eval"]["iou_thresholds"]))
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        print(f"[ep {epoch:3d}] val: " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

        # Save
        metric_val = metrics.get(save_key, metrics.get(f"{save_key}", -1))
        if metric_val > best:
            best = metric_val
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "config": cfg,
            }, out / "best.pth")
            print(f"[ep {epoch:3d}] saved new best ({save_key}={metric_val:.4f})")

        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": cfg,
        }, out / "last.pth")

    writer.close()


if __name__ == "__main__":
    main()
