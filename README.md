# VFI-Similarity

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/VFI-Similarity/blob/main/VFI_Similarity_Demo.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Video Frame Interpolation Quality Assessment via Deep Embeddings**

A tool for measuring visual similarity between video frames using fine-tuned DINOv2 embeddings. Originally designed for evaluating video frame interpolation (VFI) quality and detecting visual artifacts like flickering, tearing, and temporal inconsistencies.

<p align="center">
  <img src="docs/demo_visualization.png" alt="Demo Visualization" width="800"/>
</p>

## 🎯 Features

- **Cosine Distance Computation**: Quantify visual similarity between image pairs
- **Dense Patch Analysis**: Identify regions of difference at patch level
- **Multiple Visualizations**:
  - Similarity heatmaps (A→B and B→A)
  - Mismatch maps highlighting novel/different content
  - Vector field showing patch correspondences
  - PCA-based feature colorization
- **Batch Processing**: Process multiple pairs via JSON configuration
- **Easy Integration**: Simple Python API for custom pipelines

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- timm
- torchvision
- PIL
- matplotlib
- numpy

### Install

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/VFI-Similarity.git
cd VFI-Similarity

# Install dependencies
pip install -r requirements.txt

# Or install directly
pip install torch torchvision timm pillow matplotlib numpy scikit-learn
```

## 🚀 Quick Start

### Command Line

```bash
# Basic cosine distance between two images
python vfi_similarity.py --image-a samples/frame1.png --image-b samples/frame2.png

# With visualization output
python vfi_similarity.py --image-a samples/frame1.png --image-b samples/frame2.png \
    --visualize --out ./results

# Using custom checkpoint
python vfi_similarity.py --image-a samples/frame1.png --image-b samples/frame2.png \
    --checkpoint path/to/VFI_merged_no_lora.pt --visualize --out ./results
```

### Python API

```python
from vfi_similarity import load_model, compute_cosine_distance, run_visualization, pick_device

# Load model
device = pick_device("auto")
model = load_model(checkpoint_path="VFI_merged_no_lora.pt", device=device)

# Compute cosine distance
result = compute_cosine_distance(
    model,
    image_a="samples/frame1.png",
    image_b="samples/frame2.png",
    device=device
)
print(f"Cosine Distance: {result['cosine_distance']:.4f}")
print(f"Cosine Similarity: {result['cosine_similarity']:.4f}")

# Generate visualizations
vis_result = run_visualization(
    model,
    image_a="samples/frame1.png",
    image_b="samples/frame2.png",
    output_dir="./output",
    device=device,
    enable_pca=True
)
```

### Batch Processing

Create a JSON configuration file:

```json
[
  {
    "image_a": "samples/frame1.png",
    "image_b": "samples/frame2.png",
    "label": "pair_1"
  },
  {
    "image_a": "samples/frame3.png",
    "image_b": "samples/frame4.png",
    "label": "pair_2"
  }
]
```

Run batch processing:

```bash
python vfi_similarity.py --config pairs.json --out ./batch_results --visualize
```

## 📊 Output

### Cosine Distance

The primary output is the **cosine distance** between image embeddings:
- `0.0` = Identical images
- `1.0` = Completely different (orthogonal embeddings)
- `2.0` = Opposite (anti-correlated, rare in practice)

Typical values for VFI evaluation:
- `< 0.05`: Very similar frames (good interpolation)
- `0.05 - 0.15`: Minor differences (acceptable)
- `0.15 - 0.30`: Noticeable artifacts
- `> 0.30`: Significant visual differences

### Visualization Outputs

When `--visualize` is enabled:

| File | Description |
|------|-------------|
| `similarity_A2B.png` | Patch-wise cosine similarity from A to B |
| `similarity_B2A.png` | Patch-wise cosine similarity from B to A |
| `mismatch_A.png` | Regions in A with no good match in B |
| `mismatch_B.png` | Regions in B with no good match in A |
| `vector_field.png` | Arrows showing patch correspondences |
| `pca_features.png` | PCA-based dense feature colorization |
| `summary.json` | Statistics and metadata |

## 🏗️ Model Architecture

The model is based on **DINOv2** (ViT-B/14) fine-tuned with LoRA adapters for video frame similarity assessment. The released weights have LoRA merged into the base model for efficient inference.

```
Input Image (518×518)
        ↓
  DINOv2 ViT-B/14
        ↓
  Patch Tokens (37×37 = 1369 patches)
        ↓
  Mean Pooling → 768-dim embedding
        ↓
  Cosine Similarity/Distance
```

## 📁 Project Structure

```
VFI-Similarity/
├── vfi_similarity.py      # Main module
├── requirements.txt       # Dependencies
├── README.md             # This file
├── samples/              # Example images
│   ├── frame1.png
│   └── frame2.png
├── VFI_Similarity_Demo.ipynb  # Colab notebook
└── docs/
    └── demo_visualization.png
```

## 🔧 CLI Options

```
usage: vfi_similarity.py [-h] [--image-a IMAGE_A] [--image-b IMAGE_B]
                         [--config CONFIG] [--checkpoint CHECKPOINT]
                         [--model-name MODEL_NAME] [--resize RESIZE]
                         [--device {auto,cpu,cuda,mps}] [--visualize]
                         [--out OUT] [--arrow-stride ARROW_STRIDE] [--no-pca]

Options:
  --image-a          Path to first image (reference)
  --image-b          Path to second image (comparison)
  --config           JSON config file with image pairs
  --checkpoint       Path to merged model checkpoint
  --model-name       Base model name (default: vit_base_patch14_dinov2.lvd142m)
  --resize           Image resize dimension (default: 518)
  --device           Device: auto, cpu, cuda, mps (default: auto)
  --visualize        Enable dense patch visualization
  --out              Output directory (default: ./output)
  --arrow-stride     Stride for vector field arrows (default: 4)
  --no-pca           Disable PCA visualization
```

## 📓 Google Colab

Try it directly in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/VFI-Similarity/blob/main/VFI_Similarity_Demo.ipynb)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{vfi_similarity,
  title = {VFI-Similarity: Video Frame Interpolation Quality Assessment},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/VFI-Similarity}
}
```

## 🙏 Acknowledgments

- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI Research
- [timm](https://github.com/huggingface/pytorch-image-models) by Ross Wightman

