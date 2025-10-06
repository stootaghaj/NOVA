# NOVA: Non-Aligned Reference Image Quality Assessment for Novel View Synthesis 
### The code will be uploaded soon!

## Mismatch Overlay (B vs A)

This is a minimal CLI that computes **only** `mismatch_B_norm_overlay.png` where:
- **A** is the *reference* image
- **B** is the *processed* image

Download the weights from the link below:
```
TBA
```
The overlay is computed from DINOv2 ViT patch embeddings with a lightweight LoRA injection on the last blocks, dense nearest-neighbor correspondences, and a global-normalized mismatch heatmap **blended over image B**.

---

## Quickstart

```bash
# 1) Create env (recommended)
python -m venv .venv && source .venv/bin/activate  # (Linux/macOS)
# On Windows: .venv\Scripts\activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Run (base timm weights, no checkpoint)
python mismatch_overlay.py \
  --ref path/to/A.png \
  --proc path/to/B.png \
  --out ./outputs \
  --no-checkpoint

# 4) Or run with your fine-tuned checkpoint
python mismatch_overlay.py \
  --ref path/to/A.png \
  --proc path/to/B.png \
  --out ./outputs \
  --checkpoint /path/to/your.ckpt
```

The output will be saved as:
```
./outputs/mismatch_B_norm_overlay.png
```

---

## Arguments

- `--ref` (required): Path to **A** (reference) image.
- `--proc` (required): Path to **B** (processed) image.
- `--out` (required): Output directory.
- `--checkpoint` (optional): Path to a fine-tuned checkpoint to load on top of the timm DINOv2 model.
- `--no-checkpoint` (flag): Ignore the checkpoint and use only the base DINOv2 weights.
- `--model` (default: `vit_base_patch14_dinov2.lvd142m`): timm model name.
- `--lora-r` (default: `8`): LoRA rank injected into the last 4 transformer blocks.
- `--size` (default: `518`): Inference resize; will be adjusted to a multiple of the model patch size (14).

---

## Relation to the NOVA paper (Section **9.2**)

This code implements the **Dense Patch Matching via Nearest‑Neighbor Cosine Similarity** procedure described in Section **9.2** of the NOVA paper on Non‑Aligned Reference IQA for Novel View Synthesis. In our CLI, **A** is the *reference* image and **B** is the *processed* image. We produce a single output:
```
mismatch_B_norm_overlay.png  # global‑normalized mismatch map for B, overlaid on the original B
```


### Citation

> **NOVA — Non‑Aligned Reference Image Quality Assessment for Novel View Synthesis**, Sec. 9.2 “Dense Patch Matching via Nearest‑Neighbor Cosine Similarity”.

```bibtex
@inproceedings{NOVA2025,
  title     = {Non-Aligned Reference Image Quality Assessment for Novel View Synthesis (NOVA)},
  year      = {2025},
  note      = {See Section 9.2 for dense patch matching.}
}
```

---
