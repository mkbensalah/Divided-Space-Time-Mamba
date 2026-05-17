"""Self-supervised MAE pretraining for DST-Mamba.

Usage:
    python scripts/pretrain_mae.py \
        --config configs/pretrain_mae.yaml \
        --data_dirs data/chusj/videos data/synthetic_clips \
        --output_dir runs/pretrain
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

# Make project root importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.pretrain_dataset import PretrainClipDataset, pretrain_collate
from models import DSTMamba, DSTMambaMAE


def cosine_schedule(step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_dirs", nargs="+", required=True,
                   help="One or more directories containing pretraining videos.")
    p.add_argument("--output_dir", required=True)
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

    # Dataset
    ds = PretrainClipDataset(
        video_dirs=args.data_dirs,
        num_frames=cfg["data"]["num_frames"],
        img_size=cfg["data"]["img_size"],
        temporal_stride=cfg["data"]["temporal_stride"],
        use_depth=cfg["data"]["use_depth"],
    )
    print(f"[pretrain] {len(ds)} clips total")

    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=pretrain_collate,
        persistent_workers=cfg["train"]["num_workers"] > 0,
    )

    # Model
    encoder = DSTMamba(**cfg["model"], cls_token=False)
    model = DSTMambaMAE(
        encoder=encoder,
        decoder_dim=cfg["mae"]["decoder_dim"],
        decoder_depth=cfg["mae"]["decoder_depth"],
        in_chans=cfg["model"]["in_chans"],
        norm_target=cfg["mae"]["norm_target"],
    ).to(device)
    print(f"[pretrain] params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Optimizer (lr scaled by batch_size / 256)
    base_lr = cfg["optim"]["lr"] * cfg["train"]["batch_size"] / 256.0
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=cfg["optim"]["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"])

    steps_per_epoch = len(loader) // cfg["train"]["accum_steps"]
    total_steps = steps_per_epoch * cfg["optim"]["epochs"]
    warmup_steps = steps_per_epoch * cfg["optim"]["warmup_epochs"]

    start_epoch = 0
    global_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        print(f"[pretrain] resumed from {args.resume} (epoch {start_epoch})")

    mask_ratio = cfg["mae"]["mask_ratio"]
    log_interval = cfg["train"]["log_interval"]
    save_every = cfg["train"]["save_interval_epochs"]

    for epoch in range(start_epoch, cfg["optim"]["epochs"]):
        model.train()
        t0 = time.time()
        running = 0.0
        for it, batch in enumerate(loader):
            clip = batch["clip"].to(device, non_blocking=True)

            lr = cosine_schedule(global_step, total_steps, warmup_steps, base_lr)
            for g in optim.param_groups:
                g["lr"] = lr

            with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"]):
                out_dict = model(clip, mask_ratio=mask_ratio)
                loss = out_dict["loss"] / cfg["train"]["accum_steps"]
            scaler.scale(loss).backward()

            if (it + 1) % cfg["train"]["accum_steps"] == 0:
                if cfg["train"]["grad_clip"]:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                global_step += 1

            running += loss.item() * cfg["train"]["accum_steps"]
            if (it + 1) % log_interval == 0:
                avg = running / log_interval
                print(f"[ep {epoch:4d}][it {it:5d}/{len(loader)}] "
                      f"loss={avg:.4f}  lr={lr:.2e}  "
                      f"({(time.time() - t0) / (it + 1) * 1000:.0f} ms/iter)")
                writer.add_scalar("train/loss", avg, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                running = 0.0

        if (epoch + 1) % save_every == 0 or epoch + 1 == cfg["optim"]["epochs"]:
            ckpt_path = out / f"checkpoint-{epoch + 1:04d}.pth"
            torch.save({
                "model": model.state_dict(),
                "encoder": encoder.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "config": cfg,
            }, ckpt_path)
            # Also write a "last" pointer
            torch.save({
                "model": model.state_dict(),
                "encoder": encoder.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "config": cfg,
            }, out / "checkpoint-last.pth")
            print(f"[pretrain] saved {ckpt_path}")

    writer.close()


if __name__ == "__main__":
    main()
