#!/usr/bin/env python3
"""
NOVA: Non-aligned View Assessment for Novel View Synthesis

This tool computes cosine distance between image pairs using a fine-tuned DINOv2 model
for evaluating Novel View Synthesis (NVS) quality using non-aligned reference views.

Features:
- Cosine distance computation between image pairs
- Non-aligned reference support (no pixel-level alignment required)
- Heatmap overlay visualization on synthesized frame
- Support for batch processing via JSON config

Usage:
    # Basic quality assessment with fine-tuned checkpoint
    python nova.py --image-a reference.png --image-b synthesized.png \\
        --checkpoint weights/NOVA_merged.pt

    # With heatmap visualization
    python nova.py --image-a reference.png --image-b synthesized.png \\
        --checkpoint weights/NOVA_merged.pt --visualize --out ./results

    # Batch processing
    python nova.py --config pairs.json --out ./results \\
        --checkpoint weights/NOVA_merged.pt
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


warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Model Definition (Merged - No LoRA)
# ============================================================================

class NOVAModel(nn.Module):
    """
    NOVA: Non-aligned View Assessment model for Novel View Synthesis quality assessment.
    Built on DINOv2 with LoRA fine-tuning (merged into base model for inference).
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
               device: torch.device = None) -> NOVAModel:
    """
    Load the NOVA model for NVS quality assessment.

    Args:
        checkpoint_path: Path to merged model checkpoint (required for fine-tuned weights)
        model_name: Base model name for timm
        device: Target device

    Returns:
        Loaded model in eval mode

    Note:
        Without a checkpoint, this uses the pretrained DINOv2 weights which will
        give different results than the fine-tuned NOVA model.
    """
    if device is None:
        device = pick_device("auto")

    model = NOVAModel(model_name=model_name)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading fine-tuned checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if "base" in checkpoint:
            # Merged model format
            model.base.load_state_dict(checkpoint["base"])
            if "embedding_head" in checkpoint:
                model.embedding_head.load_state_dict(checkpoint["embedding_head"])
            print("[INFO] Loaded NOVA fine-tuned weights successfully!")
        else:
            # Try direct load
            model.load_state_dict(checkpoint, strict=False)
            print("[INFO] Loaded checkpoint (direct format)")
    else:
        print("[WARN] No checkpoint provided or file not found.")
        print("[WARN] Using pretrained DINOv2 weights - results will differ from fine-tuned NOVA model!")

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
def compute_cosine_distance(model: NOVAModel,
                            image_a: str,
                            image_b: str,
                            device: torch.device,
                            resize: int = 518) -> Dict[str, Any]:
    """
    Compute cosine distance between two images.

    Args:
        model: Loaded NOVAModel
        image_a: Path to reference image (can be non-aligned)
        image_b: Path to synthesized/distorted image
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
def extract_patch_features(model: NOVAModel,
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


def save_heatmap_overlay(values: torch.Tensor,
                         grid_size: int,
                         image: Image.Image,
                         output_path: str,
                         title: str = "",
                         cmap: str = "hot",
                         alpha: float = 0.5):
    """Save a heatmap overlayed on the original image, preserving original size and aspect ratio."""
    # Get original image size
    orig_width, orig_height = image.size

    # Create heatmap and resize to original image size
    heatmap = values.view(grid_size, grid_size).cpu().numpy()

    # Normalize heatmap to 0-1
    heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
    if heatmap_max - heatmap_min > 1e-8:
        heatmap_norm = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap_norm = heatmap

    # Resize heatmap to original image size (preserving aspect ratio)
    heatmap_resized = np.array(Image.fromarray((heatmap_norm * 255).astype(np.uint8)).resize(
        (orig_width, orig_height), Image.BILINEAR)) / 255.0

    # Calculate figure size to match aspect ratio (with reasonable max size)
    max_fig_size = 12
    aspect_ratio = orig_width / orig_height
    if aspect_ratio >= 1:
        fig_width = max_fig_size
        fig_height = max_fig_size / aspect_ratio
    else:
        fig_height = max_fig_size
        fig_width = max_fig_size * aspect_ratio

    # Create figure with correct aspect ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Show original image (not resized)
    ax.imshow(image)

    # Overlay heatmap with transparency
    heatmap_colored = ax.imshow(heatmap_resized, cmap=cmap, alpha=alpha, vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(heatmap_colored, ax=ax, shrink=0.8)
    cbar.set_label('Difference Intensity', fontsize=10)

    ax.set_title(title, fontsize=12)
    ax.axis("off")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


@torch.no_grad()
def run_visualization(model: NOVAModel,
                      image_a: str,
                      image_b: str,
                      output_dir: str,
                      device: torch.device,
                      resize: int = 518,
                      alpha: float = 0.5) -> Dict[str, Any]:
    """
    Run visualization pipeline - generates heatmap overlay on the synthesized frame.

    Args:
        model: Loaded NOVA model
        image_a: Path to reference image (can be non-aligned)
        image_b: Path to synthesized image (heatmap will be overlayed on this)
        output_dir: Output directory
        device: Computation device
        resize: Image resize dimension
        alpha: Heatmap overlay transparency (0-1)

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

    # Compute mismatch/difference map for the distorted frame (B)
    print("[INFO] Computing difference map...")
    mismatch_a, mismatch_b = compute_mismatch_maps(sim_a2b, sim_b2a, idx_a2b, idx_b2a)

    # Load distorted image for overlay
    img_b_pil = Image.open(image_b).convert("RGB")

    # Save heatmap overlay on distorted frame
    output_path = os.path.join(output_dir, "heatmap_overlay.png")
    save_heatmap_overlay(
        mismatch_b,
        grid_size,
        img_b_pil,
        output_path,
        title=f"Difference Heatmap - {os.path.basename(image_b)}",
        cmap="hot",
        alpha=alpha
    )
    print(f"[INFO] Saved heatmap overlay to: {output_path}")

    # Summary statistics
    summary = {
        "image_a": os.path.basename(image_a),
        "image_b": os.path.basename(image_b),
        "grid_size": grid_size,
        "num_patches": patches_a.size(0),
        "embedding_dim": patches_a.size(1),
        "mean_similarity_a2b": float(sim_a2b.mean()),
        "mean_similarity_b2a": float(sim_b2a.mean()),
        "mean_mismatch": float(mismatch_b.mean()),
        "output_file": output_path,
    }

    # Save summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================================
# CLI Interface
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="NOVA: Non-aligned View Assessment for Novel View Synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic quality assessment
  python nova.py --image-a reference.png --image-b synthesized.png
  
  # With heatmap visualization overlay on synthesized frame
  python nova.py --image-a reference.png --image-b synthesized.png --visualize --out ./results
  
  # Batch processing with JSON config
  python nova.py --config pairs.json --out ./results
"""
    )

    # Input options
    parser.add_argument("--image-a", type=str, help="Path to reference image (can be non-aligned)")
    parser.add_argument("--image-b", type=str, help="Path to synthesized/distorted image")
    parser.add_argument("--config", type=str, help="JSON config file with image pairs")

    # Model options
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to NOVA model checkpoint (weights/NOVA_merged.pt)")
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
                       help="Enable heatmap overlay visualization on synthesized frame")
    parser.add_argument("--out", type=str, default="./output",
                       help="Output directory for results")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Heatmap overlay transparency (0-1)")

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

        # Optional visualization (heatmap overlay only)
        if args.visualize:
            pair_out = os.path.join(args.out, f"pair_{i+1:03d}")
            vis_result = run_visualization(
                model, img_a, img_b, pair_out, device,
                args.resize, args.alpha
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

