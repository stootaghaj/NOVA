# Mismatch Overlay (B vs A)

This is a minimal CLI that computes **only** `mismatch_B_norm_overlay.png` where:
- **A** is the *reference* image
- **B** is the *processed* image

Download the weights from the link below:
```
TBD
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

### What this script does (matching the Sec. 9.2 method)
1. **Patch embeddings** – Both A and B are resized to a ViT input size, split into a g×g patch grid, and passed through a LoRA‑augmented DINOv2 to obtain L2‑normalized patch embeddings.
2. **Similarity matrix** – We compute cosine similarities between all patch pairs.
3. **Best matches (B→A and A→B)** – For each patch we take the best match in the other image.
4. **Mismatch score** – We convert similarity to mismatch via inversion and **optionally boost** non‑reciprocal matches.
5. **Global normalization** – We pool mismatch from both directions, then min–max normalize, and **use the normalized B direction** for the final overlay.
6. **Heatmap overlay** – We reshape the B‑mismatch values to g×g, upsample to the original B resolution, color‑map, and blend over B.

### Variable mapping (paper → code)
- **Patch embeddings:** `extract_tokens(...)` → `pA`, `pB`, `grid`
- **Similarity & indices:** `dense_correspondence(...)` → `sim_A2B`, `sim_B2A`, `idx_A2B`, `idx_B2A`
- **Mismatch:** `compute_mismatch(...)` → `mismatch_A`, `mismatch_B` (with optional non‑reciprocal boost)
- **Global normalization:** `mismatch_B_norm = (mismatch_B - mn) / (mx - mn)` where `mn,mx` are from concatenated A/B mismatch
- **Final artifact:** `save_overlay (origB, mismatch_B_norm, grid, "mismatch_B_norm_overlay.png", normalize=False)`

### Citation
If you use this repo in academic work, please cite the NOVA paper and reference **Section 9.2** for the dense patch matching procedure.

> **NOVA — Non‑Aligned Reference Image Quality Assessment for Novel View Synthesis**, Sec. 9.2 “Dense Patch Matching via Nearest‑Neighbor Cosine Similarity”.

```bibtex
@inproceedings{NOVA2025,
  title     = {Non-Aligned Reference Image Quality Assessment for Novel View Synthesis (NOVA)},
  year      = {2025},
  note      = {See Section 9.2 for dense patch matching.}
}
```

---
