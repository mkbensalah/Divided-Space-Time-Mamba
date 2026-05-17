# Divided Space–Time Mamba (DST-Mamba)

[![Paper](https://img.shields.io/badge/Paper-Life%202025-blue)](https://doi.org/10.3390/life15111706)
[![arXiv](https://img.shields.io/badge/DOI-10.3390%2Flife15111706-red)](https://doi.org/10.3390/life15111706)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Official implementation** of:

> Mohamed Khalil Ben Salah, Philippe Jouvet, Rita Noumeir.
> **"PICU Face and Thoracoabdominal Detection Using Self-Supervised Divided Space–Time Mamba."**
> *Life*, 15(11), 1706, 2025.
> [https://doi.org/10.3390/life15111706](https://doi.org/10.3390/life15111706)

DST-Mamba is a State Space Model–based detector for face and thoracoabdominal regions in pediatric intensive care unit (PICU) video. It factorizes spatiotemporal modeling into a spatial Bi-Mamba stage followed by a temporal Bi-Mamba stage, predicts oriented bounding boxes (OBBs), and is pretrained with masked autoencoders on domain-specific clips. The model achieves 0.96 mAP@0.5 / 0.62 mAP50-95 / 0.95 rotated IoU at 23 FPS (43 ms latency) on 16×640² clips.

## Key features

- **Divided Space–Time Mamba backbone.** Sequential spatial-then-temporal Bi-directional Mamba blocks with linear-time complexity O(L), avoiding the quadratic cost of joint self-attention.
- **Oriented bounding box detection head.** Class, axis-aligned bbox, and Circular Spatial Layout (CSL) angle prediction in 180 bins, supervised with a combined BCE + CSL + L1 + rotated IoU loss.
- **Self-supervised pretraining.** VideoMAE-style tube masking at 80% ratio on 50k+ domain clips (CHU Sainte-Justine PICU + synthetic clips generated from public pediatric/NICU images).
- **RGB-D support.** Four-channel input via early channel concatenation, with a shared patch embedding.
- **5.7× faster than YOLOv8 frame-wise and 1.9× faster than VideoMAE** at higher accuracy.

## Repository structure

```
Divided-Space-Time-Mamba/
├── configs/                  YAML configs for pretraining and fine-tuning
├── models/
│   ├── mamba_block.py        Bidirectional Mamba block (Fig. 2)
│   ├── dst_mamba.py          Divided Space-Time backbone (Fig. 1)
│   ├── patch_embed.py        2D/3D patch embedding (RGB or RGB-D)
│   ├── detection_head.py     OBB detection head (class + bbox + angle)
│   └── mae.py                MAE pretraining wrapper with tube masking
├── losses/
│   ├── detection_loss.py     Combined detection loss
│   └── rotated_iou.py        Differentiable rotated IoU
├── data/
│   ├── video_obb_dataset.py  General video OBB dataset (any annotated collection)
│   ├── synthetic_video.py    Image-to-video clip generation
│   └── transforms.py         Spatiotemporal augmentations
├── utils/
│   ├── csl.py                Circular Spatial Layout encoding/decoding
│   ├── metrics.py            mAP, rIoU, temporal IoU, angle accuracy
│   └── obb.py                Oriented bounding box utilities
├── scripts/
│   ├── pretrain_mae.py       MAE self-supervised pretraining
│   ├── finetune.py           Detection fine-tuning
│   ├── evaluate.py           Test-set evaluation
│   └── visualize.py          Qualitative detection visualization
├── requirements.txt
└── README.md
```

## Installation

Tested on Ubuntu 22.04, Python 3.10, CUDA 11.8, PyTorch 2.0.1, NVIDIA Tesla V100S-PCIE-32GB.

```bash
git clone https://github.com/mkbensalah/Divided-Space-Time-Mamba.git
cd Divided-Space-Time-Mamba

# Create environment
conda create -n dstmamba python=3.10 -y
conda activate dstmamba

# PyTorch (match your CUDA)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Mamba kernels (required)
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.1.1

# Everything else
pip install -r requirements.txt
```

> The `mamba-ssm` and `causal-conv1d` packages require a CUDA-capable GPU and a recent gcc. If installation fails, see their respective troubleshooting notes.

## Data preparation

### Dataset layout

`VideoOBBDataset` works with any annotated video collection. Organize your data as:

```
data/my_dataset/
├── videos/
│   ├── video_001.mp4
│   ├── video_001_depth.mp4   # optional, for RGB-D
│   └── ...
├── annotations/
│   └── video_001.json        # OBB per frame: see data/example_annotation.json
└── splits/
    ├── train.txt             # one video_id per line
    ├── val.txt
    └── test.txt
```

See `data/example_annotation.json` for the full annotation schema.

### CHU Sainte-Justine PICU dataset
The clinical dataset used in the paper is from the MEDEVAC database (CHU Sainte-Justine, ethics protocol 2016-1242). It is not publicly redistributable; access requires Research Ethics Board approval. Contact the corresponding author.

### Synthetic pretraining clips
Generate video clips from publicly available pediatric/NICU images:
```bash
python -m data.synthetic_video \
    --input_dir /path/to/public_images \
    --output_dir data/synthetic_clips \
    --num_frames 16 \
    --target_clips 15000
```

## Training

### Self-supervised MAE pretraining
```bash
python scripts/pretrain_mae.py \
    --config configs/pretrain_mae.yaml \
    --data_dirs data/my_dataset/videos data/synthetic_clips \
    --output_dir runs/pretrain
```

Defaults follow the paper: 2500 epochs, AdamW (lr=1.5e-4, wd=0.05, warmup=40), 80% tube masking, encoder depth 12 / dim 768, decoder depth 4 / dim 384, 16-frame 224² clips.

### Fine-tuning for OBB detection
```bash
python scripts/finetune.py \
    --config configs/finetune_detection.yaml \
    --pretrained runs/pretrain/checkpoint-last.pth \
    --data_root data/my_dataset \
    --output_dir runs/finetune
```

Defaults: 100 epochs, RAdam (lr=1e-3), batch size 32, 16-frame clips at 224² or 640², combined loss (BCE + CSL + α·L1 + β·rotated IoU).

### Evaluation
```bash
python scripts/evaluate.py \
    --checkpoint runs/finetune/best.pth \
    --data_root data/my_dataset \
    --split test
```

Reports rIoU, mAP@{0.5, 0.6, 0.75}, mAP50-95, angle accuracy, temporal IoU.

### Visualization
```bash
python scripts/visualize.py \
    --checkpoint runs/finetune/best.pth \
    --data_root data/my_dataset \
    --videos video_001 video_002 video_003 \
    --output_dir figures/qualitative
```

Renders ground-truth (solid) and predicted (dashed) oriented bounding boxes on sampled frames, and saves a combined PDF.

## Reproducing paper results

| Component | Value |
|---|---|
| Backbone | DST-Mamba, depth=12, embed_dim=768 |
| Patch | 16×16 spatial, tubelet=1 |
| Input | 16 frames × 224² (or 640²) |
| Pretraining | MAE, mask=80%, epochs=2500, lr=1.5e-4 (AdamW) |
| Fine-tuning | epochs=100, lr=1e-3 (RAdam), batch=32 |
| Loss | BCE + CSL + α·L1 + β·rotated IoU (α=1.0, β=2.0) |
| Hardware | 1× Tesla V100S 32GB |

Reported headline numbers on the CHU-SJ test split (112 patients):

| Metric | Value |
|---|---|
| mAP@0.5 | 0.96 |
| mAP50-95 | 0.62 |
| rIoU | 0.95 |
| Temporal IoU | 0.95 |
| Latency (16×640²) | 43 ms / 23 FPS |
| FLOPs (16×640²) | 7.56 G |
| Parameters | 73 M |

## Citation

```bibtex
@article{bensalah2025dstmamba,
  title   = {PICU Face and Thoracoabdominal Detection Using Self-Supervised Divided Space-Time Mamba},
  author  = {Ben Salah, Mohamed Khalil and Jouvet, Philippe and Noumeir, Rita},
  journal = {Life},
  volume  = {15},
  number  = {11},
  pages   = {1706},
  year    = {2025},
  doi     = {10.3390/life15111706}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

This work was supported by NSERC (Canada) and FRQS. Clinical data collection at CHU Sainte-Justine was approved under ethics protocol 2016-1242. We thank the PICU staff and families who consented to data collection.

Implementation builds on prior open-source work: [Mamba](https://github.com/state-spaces/mamba), [VideoMAE](https://github.com/MCG-NJU/VideoMAE), [VideoMamba](https://github.com/OpenGVLab/VideoMamba), [Unmasked Teacher](https://github.com/OpenGVLab/unmasked_teacher), and [TimeSformer](https://github.com/facebookresearch/TimeSformer).
