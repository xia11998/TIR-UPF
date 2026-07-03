# Dataset Placeholder

Do not commit datasets to this repository.

Expected paired training layout:

```text
datasets/M3FD_Detection/
  train/
    input/
    target/
    degradation_labels.npy  # optional
  test/
    input/
    target/
```

For custom inference, pass any image folder to `test_custom.py` with `--dataset_dir`.
