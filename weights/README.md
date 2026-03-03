# Model Weights

This directory contains pre-trained NOVA model checkpoints.

## Available Checkpoints

| Checkpoint | Training Data | Description |
|------------|--------------|-------------|
| `NOVA_baseline.pt` | Synthetic distortions | Trained on synthetically distorted data. Reproduces the results reported in the paper. |
| `NOVA_NVS.pt` | NVS distortions | Fine-tuned on real NVS artifacts. **Recommended for NVS quality assessment.** |

## Download

Model weights are included via [Git LFS](https://git-lfs.github.com/) and downloaded automatically when you clone the repository.

If the weights weren't downloaded, run:

```bash
git lfs install
git lfs pull
```

For more details and dataset, visit our [project page](https://stootaghaj.github.io/nova-project/).

## Usage

```bash
# Using baseline checkpoint (paper results)
python nova.py --image-a samples/frame1.png --image-b samples/frame2.png \
    --checkpoint weights/NOVA_baseline.pt

# Using NVS-optimized checkpoint (recommended for NVS tasks)
python nova.py --image-a samples/frame1.png --image-b samples/frame2.png \
    --checkpoint weights/NOVA_NVS.pt
```
