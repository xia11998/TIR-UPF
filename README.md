# TIR-UPF
A Unified Progressive Fusion Network with Adaptive Degradation Prompts for Multi-Weather Thermal Infrared Image Restoration

Source-code archive version: **1.0.0**. The accompanying manuscript is under revision.

The implementation includes a four-class degradation predictor, dynamic prompt fusion, a prompt-conditioned restoration backbone, and three restoration stages. This archive corrects runtime interfaces and usage instructions while retaining the existing model architecture and training defaults. Dataset images and trained checkpoints are not bundled.

Source repository: https://github.com/xia11998/TIR-UPF

## Included source files

- `train.py`: training entry point, including single-stage and prompt-only options.
- `dataset.py`: paired train/validation data loaders.
- `inference.py` and `evaluation.py`: checkpoint-based restoration and paired-image evaluation.
- `models/`: the restoration backbone and four-class degradation predictor.
- `data/type7.npy` and `data/multi2.npy`: fixed prompt tensors required by the released implementation; they are not images, labels, or learned model weights.
- `datasets/degradation_labels.npy`: the existing filename-to-class label map; it does not contain image pixels.
- `utils/` and `warmup_scheduler/`: runtime dependencies for training and evaluation.

## Installation

Use Python 3.8-3.10; Python 3.10 is recommended for the pinned PyTorch version. Training requires a CUDA-capable GPU. Inference and evaluation use CUDA when available and otherwise run on CPU.

Install a suitable PyTorch 1.12.0 build first. For CUDA 11.6, the official installation command is:

```bash
python -m pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

For CPU-only inference/evaluation, use:

```bash
python -m pip install torch==1.12.0+cpu torchvision==0.13.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

See the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/#v1120) for other supported builds. NumPy is constrained below 2 for compatibility with this PyTorch version.

## Data Preparation

Run commands from the repository root. Supply paired grayscale images with identical filenames:

```text
datasets/M3FD_Detection/
  train/
    input/
    target/
    degradation_labels.npy
  val/
    input/
    target/
  test/
    input/
    target/
```

Use the actual data partitions from your experiment; these directory names are examples. Do not use an independent test set to select the best training checkpoint.

The training label map is a NumPy-saved dictionary of image basenames to integer class IDs. The script treats class 0 as clean and classes 1-3 as degraded; retain the exact class mapping used in your experiment. If the supplied map matches your training filenames, copy it to `train/degradation_labels.npy`. Otherwise, supply your experiment's own map. See [datasets/README.md](datasets/README.md) for supported locations and legacy fallback behavior.

## Training

```bash
python train.py --dataset_dir ./datasets/M3FD_Detection --train_split train --val_split val
```

The script defaults are batch size 16, patch size 48, 300 epochs, seed 1234, initial learning rate `8e-5`, validation every 20 epochs, and three restoration stages. These defaults are preserved; reproduce a particular manuscript experiment using its actual settings and data split. The default `--val_split` remains `test`, so explicitly select the appropriate validation split when running training.

Training writes checkpoints to `exps/TIR-UPF_M3FD_train/models/`. A full-method checkpoint includes `state_dict`, `predictor_state_dict`, and `prompt_bank`.

## Inference

Supply a checkpoint produced by the training script:

```bash
python inference.py --dataset_dir ./datasets/M3FD_Detection/test --dataset_name input --weights ./checkpoint/tir_upf.pth --output ./results
```

This reads images from `datasets/M3FD_Detection/test/input/` and saves restored images to `results/input/`. With a full-method checkpoint, the default `--prompt_mode auto` uses the learned predictor and prompt bank.

## Evaluation

```bash
python evaluation.py --input_dir ./datasets/M3FD_Detection/test/input --target_dir ./datasets/M3FD_Detection/test/target --weights ./checkpoint/tir_upf.pth --output ./results/evaluation
```

The script reports PSNR/SSIM and saves restored images. Input and target filenames and image dimensions must match. Evaluation pads images as needed by the backbone and crops outputs to their original dimensions before computing metrics.

Inspect supported options with `python train.py --help`, `python inference.py --help`, and `python evaluation.py --help`.

## Archive and License

This is a source-code archive, not a bundle of dataset images or manuscript figures. Actual experiment splits, weather-degradation generation settings, and any trained checkpoint must be supplied separately for a full reproduction of manuscript results.

For a manual Zenodo upload, upload the source ZIP as resource type **Software**, enter the software authors, version `1.0.0`, description, and MIT license, and link the GitHub repository. Alternatively, enable the repository in Zenodo before publishing the corresponding GitHub Release. Add the assigned version DOI to the manuscript only after the record is published; no DOI is preassigned in this archive.

See [LICENSE](LICENSE) and the retained source headers for upstream attribution.

