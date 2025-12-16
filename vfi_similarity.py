#!/usr/bin/env python3
"""
VFI-Similarity: Video Frame Interpolation Quality Assessment via Deep Embeddings

This tool computes cosine distance between image pairs using a fine-tuned DINOv2 model
for evaluating video frame interpolation quality and detecting visual artifacts.

Features:
- Cosine distance computation between image pairs
- Optional dense patch visualization (PCA, mismatch maps, vector fields)
- Support for batch processing via JSON config

Usage:
    # Basic cosine distance
    python vfi_similarity.py --image-a frame1.png --image-b frame2.png

    # With visualization
    python vfi_similarity.py --image-a frame1.png --image-b frame2.png --visualize --out ./results

    # Batch processing
    python vfi_similarity.py --config pairs.json --out ./results
"""

import os
import sys
import json
import math
import argparse
import warnings
from typing import Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    timm = None
    print("[WARN] timm not installed. Install with: pip install timm")

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    PCA = None

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Model Definition (Merged - No LoRA)
# ============================================================================

class VFISimilarityModel(nn.Module):
    """
    DINOv2-based model for video frame similarity assessment.
    This is the merged version where LoRA weights have been baked into the base model.
    """
    def __init__(self, model_name: str = "vit_base_patch14_dinov2.lvd142m"):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm is required. Install with: pip install timm")

        self.base = timm.create_model(model_name, pretrained=True)
        self.num_features = self.base.num_features
        self.base.head = nn.Identity()

        # Projection head (optional, for future use)
        self.embedding_head = nn.Sequential(
            nn.Linear(self.num_features, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.LayerNorm(512)
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch features from input image."""
        return self.base.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base model."""
        return self.base(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get mean-pooled embedding from patch features."""
        feat = self.forward_features(x)
        return torch.mean(feat, dim=1)


# ============================================================================
# Model Loading
# ============================================================================

def load_model(checkpoint_path: Optional[str] = None,
               model_name: str = "vit_base_patch14_dinov2.lvd142m",
               device: torch.device = None) -> VFISimilarityModel:
    """
    Load the VFI similarity model.

    Args:
        checkpoint_path: Path to merged model checkpoint (optional)
        model_name: Base model name for timm
        device: Target device

    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = pick_device("auto")

    model = VFISimilarityModel(model_name=model_name)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if "base" in checkpoint:
            # Merged model format
            model.base.load_state_dict(checkpoint["base"])
            if "embedding_head" in checkpoint:
                model.embedding_head.load_state_dict(checkpoint["embedding_head"])
        else:
            # Try direct load
            model.load_state_dict(checkpoint, strict=False)
    else:
        print("[INFO] Using pretrained DINOv2 weights (no fine-tuned checkpoint)")

    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# Device Selection
# ============================================================================

def pick_device(choice: str = "auto") -> torch.device:
    """Select the best available device."""
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# Image Preprocessing
# ============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(size: int = 518):
    """Build image transform pipeline."""
    return transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_image(path: str, transform=None) -> torch.Tensor:
    """Load and preprocess an image."""
    img = Image.open(path).convert("RGB")
    if transform is None:
        transform = build_transform()
    return transform(img).unsqueeze(0)


# ============================================================================
# Core Similarity Computation
# ============================================================================

@torch.no_grad()
def compute_cosine_distance(model: VFISimilarityModel,
                            image_a: str,
                            image_b: str,
                            device: torch.device,
                            resize: int = 518) -> Dict[str, Any]:
    """
    Compute cosine distance between two images.

    Args:
        model: Loaded VFISimilarityModel
        image_a: Path to first image (reference)
        image_b: Path to second image (comparison)
        device: Computation device
        resize: Image resize dimension

    Returns:
        Dictionary with cosine_distance, cosine_similarity, and metadata
    """
    transform = build_transform(resize)

    # Load images
    img_a = load_image(image_a, transform).to(device)
    img_b = load_image(image_b, transform).to(device)

    # Get embeddings
    emb_a = model.get_embedding(img_a)
    emb_b = model.get_embedding(img_b)

    # Compute cosine similarity
    similarity = F.cosine_similarity(emb_a, emb_b).item()
    distance = 1.0 - similarity

    return {
        "image_a": os.path.basename(image_a),
        "image_b": os.path.basename(image_b),
        "cosine_similarity": similarity,
        "cosine_distance": distance,
        "embedding_dim": emb_a.shape[-1],
    }


# ============================================================================
# Dense Patch Analysis & Visualization
# ============================================================================

@torch.no_grad()
def extract_patch_features(model: VFISimilarityModel,
                           image_path: str,
                           device: torch.device,
                           resize: int = 518) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Extract dense patch features from an image.

    Returns:
        cls_token: CLS token embedding
        patches: Patch embeddings [N, C]
        grid_size: Spatial grid dimension
    """
    transform = build_transform(resize)
    img = load_image(image_path, transform).to(device)

    features = model.forward_features(img)

    if features.ndim == 3:
        cls_token = features[:, 0]
        patches = features[:, 1:]
    else:
        raise RuntimeError(f"Unexpected feature shape: {features.shape}")

    patches = patches[0].float().cpu()
    cls_token = cls_token[0].float().cpu()

    N = patches.size(0)
    grid_size = int(math.isqrt(N))
    if grid_size * grid_size != N:
        raise ValueError(f"Patch count {N} is not a perfect square")

    return cls_token, patches, grid_size


def dense_nearest_neighbors(patches_a: torch.Tensor,
                           patches_b: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Compute dense nearest neighbor matching between patch sets."""
    a_norm = F.normalize(patches_a, dim=-1)
    b_norm = F.normalize(patches_b, dim=-1)

    similarity_matrix = a_norm @ b_norm.t()

    sim_a2b, idx_a2b = similarity_matrix.max(dim=1)
    sim_b2a, idx_b2a = similarity_matrix.t().max(dim=1)

    return idx_a2b, sim_a2b, idx_b2a, sim_b2a


def compute_mismatch_maps(sim_a2b: torch.Tensor,
                          sim_b2a: torch.Tensor,
                          idx_a2b: torch.Tensor,
                          idx_b2a: torch.Tensor,
                          boost: float = 1.25) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mismatch maps highlighting novel/different content.
    Non-reciprocal matches are boosted to emphasize potential artifacts.
    """
    # Normalize similarities
    def normalize(x):
        vmin, vmax = x.min(), x.max()
        return (x - vmin) / max(vmax - vmin, 1e-8)

    sim_a_norm = normalize(sim_a2b)
    sim_b_norm = normalize(sim_b2a)

    # Base mismatch (inverse of similarity)
    mismatch_a = 1 - sim_a_norm
    mismatch_b = 1 - sim_b_norm

    # Boost non-reciprocal matches
    recip_a = (idx_b2a[idx_a2b] == torch.arange(len(idx_a2b)))
    recip_b = (idx_a2b[idx_b2a] == torch.arange(len(idx_b2a)))

    mismatch_a = mismatch_a * torch.where(recip_a, torch.ones_like(mismatch_a),
                                          torch.full_like(mismatch_a, boost))
    mismatch_b = mismatch_b * torch.where(recip_b, torch.ones_like(mismatch_b),
                                          torch.full_like(mismatch_b, boost))

    return mismatch_a, mismatch_b


def save_heatmap(values: torch.Tensor,
                 grid_size: int,
                 output_path: str,
                 title: str = "",
                 cmap: str = "inferno",
                 vmin: float = None,
                 vmax: float = None):
    """Save a heatmap visualization."""
    heatmap = values.view(grid_size, grid_size).cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=10)
    plt.axis("off")
    plt.colorbar(shrink=0.8)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def get_patch_centers(width: int, height: int, grid_size: int) -> np.ndarray:
    """Get pixel coordinates of patch centers."""
    xs = np.linspace(width / (2 * grid_size), width - width / (2 * grid_size), grid_size)
    ys = np.linspace(height / (2 * grid_size), height - height / (2 * grid_size), grid_size)
    xv, yv = np.meshgrid(xs, ys)
    return np.stack([xv, yv], axis=-1).reshape(-1, 2)


def draw_vector_field(image_a: Image.Image,
                      image_b: Image.Image,
                      grid_size: int,
                      idx_a2b: torch.Tensor,
                      output_path: str,
                      stride: int = 4):
    """Draw correspondence arrows between image patches."""
    a_arr = np.array(image_a)
    b_arr = np.array(image_b)
    H, W = a_arr.shape[:2]

    canvas = np.zeros((H, W * 2, 3), dtype=a_arr.dtype)
    canvas[:, :W] = a_arr
    canvas[:, W:] = b_arr

    centers_a = get_patch_centers(W, H, grid_size)
    centers_b = get_patch_centers(W, H, grid_size)
    centers_b[:, 0] += W

    plt.figure(figsize=(12, 6))
    plt.imshow(canvas)
    plt.axis("off")

    for r in range(0, grid_size, stride):
        for c in range(0, grid_size, stride):
            i = r * grid_size + c
            j = idx_a2b[i].item()
            x1, y1 = centers_a[i]
            x2, y2 = centers_b[j]
            plt.arrow(x1, y1, x2 - x1, y2 - y1,
                     head_width=6, head_length=4,
                     linewidth=0.5, alpha=0.7, color="cyan")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def visualize_pca(patches_a: torch.Tensor,
                  patches_b: torch.Tensor,
                  grid_size: int,
                  output_dir: str):
    """Create PCA-based dense colorization of patches."""
    if not HAS_SKLEARN:
        print("[WARN] sklearn not installed, skipping PCA visualization")
        return

    # Fit PCA on combined patches
    all_patches = torch.cat([patches_a, patches_b], dim=0).numpy()
    pca = PCA(n_components=3, whiten=True, random_state=42)
    pca.fit(all_patches)

    # Project each image
    proj_a = pca.transform(patches_a.numpy())
    proj_b = pca.transform(patches_b.numpy())

    # Reshape and normalize to RGB
    proj_a = proj_a.reshape(grid_size, grid_size, 3)
    proj_b = proj_b.reshape(grid_size, grid_size, 3)

    # Sigmoid scaling for visualization
    proj_a = 1 / (1 + np.exp(-2 * proj_a))
    proj_b = 1 / (1 + np.exp(-2 * proj_b))

    # Save visualizations
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(proj_a)
    axes[0].set_title("Image A - PCA Features")
    axes[0].axis("off")
    axes[1].imshow(proj_b)
    axes[1].set_title("Image B - PCA Features")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_features.png"), dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def run_visualization(model: VFISimilarityModel,
                      image_a: str,
                      image_b: str,
                      output_dir: str,
                      device: torch.device,
                      resize: int = 518,
                      arrow_stride: int = 4,
                      enable_pca: bool = True) -> Dict[str, Any]:
    """
    Run full visualization pipeline.

    Returns:
        Dictionary with similarity statistics and output paths
    """
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Extracting patch features...")
    _, patches_a, grid_a = extract_patch_features(model, image_a, device, resize)
    _, patches_b, grid_b = extract_patch_features(model, image_b, device, resize)

    if grid_a != grid_b:
        raise ValueError("Grid size mismatch between images")

    grid_size = grid_a
    print(f"[INFO] Grid: {grid_size}x{grid_size}, Patches: {patches_a.size(0)}, Dim: {patches_a.size(1)}")

    print("[INFO] Computing dense correspondences...")
    idx_a2b, sim_a2b, idx_b2a, sim_b2a = dense_nearest_neighbors(patches_a, patches_b)

    # Similarity heatmaps
    print("[INFO] Generating similarity heatmaps...")
    save_heatmap(sim_a2b, grid_size, os.path.join(output_dir, "similarity_A2B.png"),
                 "A→B Cosine Similarity", cmap="viridis")
    save_heatmap(sim_b2a, grid_size, os.path.join(output_dir, "similarity_B2A.png"),
                 "B→A Cosine Similarity", cmap="viridis")

    # Mismatch maps
    print("[INFO] Computing mismatch maps...")
    mismatch_a, mismatch_b = compute_mismatch_maps(sim_a2b, sim_b2a, idx_a2b, idx_b2a)
    save_heatmap(mismatch_a, grid_size, os.path.join(output_dir, "mismatch_A.png"),
                 "Mismatch Map A", cmap="hot")
    save_heatmap(mismatch_b, grid_size, os.path.join(output_dir, "mismatch_B.png"),
                 "Mismatch Map B", cmap="hot")

    # Vector field
    print("[INFO] Drawing correspondence vectors...")
    img_a_pil = Image.open(image_a).convert("RGB").resize((resize, resize))
    img_b_pil = Image.open(image_b).convert("RGB").resize((resize, resize))
    draw_vector_field(img_a_pil, img_b_pil, grid_size, idx_a2b,
                     os.path.join(output_dir, "vector_field.png"), stride=arrow_stride)

    # PCA visualization
    if enable_pca:
        print("[INFO] Generating PCA visualization...")
        visualize_pca(patches_a, patches_b, grid_size, output_dir)

    # Summary statistics
    summary = {
        "image_a": os.path.basename(image_a),
        "image_b": os.path.basename(image_b),
        "grid_size": grid_size,
        "num_patches": patches_a.size(0),
        "embedding_dim": patches_a.size(1),
        "mean_similarity_a2b": float(sim_a2b.mean()),
        "mean_similarity_b2a": float(sim_b2a.mean()),
        "mean_mismatch_a": float(mismatch_a.mean()),
        "mean_mismatch_b": float(mismatch_b.mean()),
        "output_dir": output_dir,
    }

    # Save summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Visualizations saved to: {output_dir}")
    return summary


# ============================================================================
# CLI Interface
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="VFI-Similarity: Video Frame Interpolation Quality Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic cosine distance
  python vfi_similarity.py --image-a frame1.png --image-b frame2.png
  
  # With visualization
  python vfi_similarity.py --image-a frame1.png --image-b frame2.png --visualize --out ./results
  
  # Batch processing with JSON config
  python vfi_similarity.py --config pairs.json --out ./results
"""
    )

    # Input options
    parser.add_argument("--image-a", type=str, help="Path to first image (reference)")
    parser.add_argument("--image-b", type=str, help="Path to second image (comparison)")
    parser.add_argument("--config", type=str, help="JSON config file with image pairs")

    # Model options
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to merged model checkpoint")
    parser.add_argument("--model-name", type=str, default="vit_base_patch14_dinov2.lvd142m",
                       help="Base model name for timm")

    # Processing options
    parser.add_argument("--resize", type=int, default=518,
                       help="Image resize dimension")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device selection")

    # Visualization options
    parser.add_argument("--visualize", action="store_true",
                       help="Enable dense patch visualization")
    parser.add_argument("--out", type=str, default="./output",
                       help="Output directory for results")
    parser.add_argument("--arrow-stride", type=int, default=4,
                       help="Stride for vector field arrows")
    parser.add_argument("--no-pca", action="store_true",
                       help="Disable PCA visualization")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Validate inputs
    if args.config:
        if not os.path.exists(args.config):
            print(f"[ERROR] Config file not found: {args.config}")
            sys.exit(1)
    elif not (args.image_a and args.image_b):
        print("[ERROR] Provide either --config or both --image-a and --image-b")
        parser.print_help()
        sys.exit(1)

    # Setup device and model
    device = pick_device(args.device)
    print(f"[INFO] Using device: {device}")

    model = load_model(args.checkpoint, args.model_name, device)

    # Process pairs
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        pairs = config if isinstance(config, list) else config.get("pairs", [])
    else:
        pairs = [{"image_a": args.image_a, "image_b": args.image_b}]

    results = []
    for i, pair in enumerate(pairs):
        img_a = pair.get("image_a") or pair.get("ref_img")
        img_b = pair.get("image_b") or pair.get("dist_img")

        if not (img_a and img_b):
            print(f"[WARN] Skipping invalid pair: {pair}")
            continue

        print(f"\n[{i+1}/{len(pairs)}] Processing: {os.path.basename(img_a)} vs {os.path.basename(img_b)}")

        # Compute cosine distance
        result = compute_cosine_distance(model, img_a, img_b, device, args.resize)
        print(f"  → Cosine Distance: {result['cosine_distance']:.4f}")
        print(f"  → Cosine Similarity: {result['cosine_similarity']:.4f}")

        # Optional visualization
        if args.visualize:
            pair_out = os.path.join(args.out, f"pair_{i+1:03d}")
            vis_result = run_visualization(
                model, img_a, img_b, pair_out, device,
                args.resize, args.arrow_stride, not args.no_pca
            )
            result.update(vis_result)

        results.append(result)

    # Save all results
    os.makedirs(args.out, exist_ok=True)
    output_file = os.path.join(args.out, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Results saved to: {output_file}")

    # Print summary
    if len(results) > 0:
        avg_dist = sum(r["cosine_distance"] for r in results) / len(results)
        print(f"\n📊 Summary: {len(results)} pair(s) processed")
        print(f"   Average Cosine Distance: {avg_dist:.4f}")


if __name__ == "__main__":
    main()

