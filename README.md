# TIR-UPF
A Unified Progressive Fusion Network with Adaptive Degradation Prompts for Multi-Weather Thermal Infrared Image Restoration

This package contains the method implementation that structurally matches the manuscript: a four-class degradation predictor, prompt-conditioned restoration backbone, dynamic prompt fusion, and a three-stage restoration configuration. The released training defaults use the manuscript-level configuration: full predictor mode, dynamic prompt modulation, classification/orthogonality/edge losses, global residual learning, and three restoration stages.

Source repository: https://github.com/xia11998/TIR-UPF

## Included source files

- `train.py`: full-method training entry point and ablation switches.
- `dataset.py`: paired train/validation data loaders.
- `inference.py` and `evaluation.py`: checkpoint-based restoration and paired-image evaluation.
- `models/`: the restoration backbone and four-class degradation predictor.
- `data/type7.npy` and `data/multi2.npy`: fixed prompt tensors required by the released implementation; they are not images, labels, or learned model weights.
- `utils/` and `warmup_scheduler/`: runtime dependencies for training and evaluation.

## Installation

```bash
pip install -r requirements.txt
```

## Training interface

```bash
python train.py \
  --train_input_dir /path/to/train/input \
  --train_target_dir /path/to/train/target \
  --val_input_dir /path/to/validation/input \
  --val_target_dir /path/to/validation/target \
  --predictor_mode full --ablation full
```
## Evaluation

```bash
python evaluation.py --input_dir ./datasets/M3FD_Detection/test/input --target_dir ./datasets/M3FD_Detection/test/target --weights ./checkpoint/tir_upf.pth --output ./results/evaluation
```

The script reports PSNR/SSIM and saves restored images. Input and target filenames and image dimensions must match. Evaluation pads images as needed by the backbone and crops outputs to their original dimensions before computing metrics.

Inspect supported options with `python train.py --help`, `python inference.py --help`, and `python evaluation.py --help`.
