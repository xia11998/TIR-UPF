# Dataset Preparation

Dataset images are not bundled with this source-code archive. The existing `degradation_labels.npy` contains filename-to-class metadata only.

Expected paired training layout:

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

Input and target images must have identical filenames and dimensions. The directory names are examples; use the experiment's actual train/validation/test partitions.

The paired training loader searches for labels in this order:

1. `<dataset_dir>/<train_split>/degradation_labels.npy`
2. `<dataset_dir>/degradation_labels.npy`
3. `<parent_of_dataset_dir>/corrupt_labeled/degradation_labels.npy`

Copy the supplied label dictionary to one of these locations only if it matches your training images and class definitions. Otherwise, provide the label map used in your own experiment. With four classes, labels must be integers in the range 0-3; the training script treats class 0 as clean.

The training loader's legacy behavior is retained: missing labels default to 0, and missing paired targets fall back to the input image. Prepare complete labels and targets for restoration experiments; these fallback cases are not substitutes for annotated paired training data.

For inference on a folder such as `datasets/example/input/`, run:

```bash
python inference.py --dataset_dir ./datasets/example --dataset_name input --weights ./checkpoint/tir_upf.pth
```

For paired evaluation, pass both `--input_dir` and `--target_dir` to `evaluation.py`.

