#!/usr/bin/env python3
"""
Pose-free evaluation script for MapAnything model.
Processes images and generates 3D reconstructions with camera poses.
"""

import argparse
import os
import sys
from pathlib import Path

# Optional config for better memory efficiency
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images


import numpy as np
from PIL import Image
import matplotlib.cm as cm

def depth_to_png(depth_t: torch.Tensor,
                 global_stats=None,      # (lo, hi) across the whole scene if you have them
                 percentile=(2, 98),     # robust per-image fallback
                 invert_if_disparity=True,
                 save_16bit=True,
                 path="outputs/depth_map_view.png"):
    """
    depth_t: [H,W] or [1,1,H,W], Z in camera coords or disparity/inverse-depth.
    """
    # to numpy
    depth = depth_t.squeeze().detach().cpu().numpy().astype(np.float32)

    # mask invalids
    valid = np.isfinite(depth) & (depth > 0)
    if valid.sum() == 0:
        raise ValueError("No valid depth values to visualize.")

    d = depth.copy()

    # If your model actually returns disparity/inverse-depth, flip it to pseudo-Z for nicer viz
    if invert_if_disparity and (np.nanmedian(d[valid]) < np.nanmedian(1.0 / np.clip(d[valid], 1e-9, None))):
        # Heuristic: many disparity maps have small values for far stuff.
        # If you *know* it’s disparity, just set d = 1.0 / d.
        pass  # keep as-is; this branch is just a placeholder for your own signal
    # If you *know* it's disparity:
    # d[valid] = 1.0 / np.clip(d[valid], 1e-6, None)

    # Choose scaling range
    if global_stats is not None:
        lo, hi = global_stats
    else:
        lo = np.percentile(d[valid], percentile[0])
        hi = np.percentile(d[valid], percentile[1])

    # clip and normalize
    d = np.clip(d, lo, hi)
    d = (d - lo) / max(hi - lo, 1e-6)

    # Optional: inverse for prettier “near = bright”
    d_vis = 1.0 - d

    # Save 16-bit gray (preserves structure)
    if save_16bit:
        img16 = (d_vis * 65535.0).astype(np.uint16)
        Image.fromarray(img16, mode="I;16").save(path.replace(".png", "_16bit.png"))

    # Also save a colormapped 8-bit for quick viewing
    cmap = cm.get_cmap("magma")  # perceptual; viewer-friendly
    rgb = (cmap(d_vis)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path.replace(".png", "_magma.png"))

    return (lo, hi)


def setup_device():
    """Setup and return the appropriate device for inference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def load_model(device, model_name="facebook/map-anything"):
    """
    Load the MapAnything model.
    
    Args:
        device: Device to load the model on
        model_name: Name of the model to load (default: "facebook/map-anything")
                   For Apache 2.0 license model, use "facebook/map-anything-apache"
    
    Returns:
        Loaded MapAnything model
    """
    print(f"Loading model: {model_name}")
    print("Note: This requires internet access or the huggingface hub cache to be pre-downloaded")
    
    model = MapAnything.from_pretrained(model_name).to(device)
    print("Model loaded successfully")
    return model


def load_images_from_folder(image_folder):
    """
    Load and preprocess images from a folder.
    
    Args:
        image_folder: Path to folder containing images
    
    Returns:
        Preprocessed views ready for inference
    """
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    
    print(f"Loading images from: {image_folder}")
    views = load_images(image_folder)
    print(f"Loaded {len(views)} images")
    return views


def run_inference(model, views, memory_efficient=False, use_amp=True, 
                 amp_dtype="bf16", apply_mask=True, mask_edges=True,
                 apply_confidence_mask=False, confidence_percentile=10):
    """
    Run inference on the loaded views.
    
    Args:
        model: Loaded MapAnything model
        views: Preprocessed image views
        memory_efficient: Trades off speed for more views (up to 2000 views on 140 GB)
        use_amp: Use mixed precision inference (recommended)
        amp_dtype: bf16 inference (recommended; falls back to fp16 if bf16 not supported)
        apply_mask: Apply masking to dense geometry outputs
        mask_edges: Remove edge artifacts by using normals and depth
        apply_confidence_mask: Filter low-confidence regions
        confidence_percentile: Remove bottom percentile confidence pixels
    
    Returns:
        List of predictions for each view
    """
    print("Running inference...")
    predictions = model.infer(
        views,                            # Input views
        memory_efficient_inference=memory_efficient,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        apply_mask=apply_mask,
        mask_edges=mask_edges,
        apply_confidence_mask=apply_confidence_mask,
        confidence_percentile=confidence_percentile,
    )
    print(f"Inference completed for {len(predictions)} views")
    return predictions


def process_predictions(predictions):
    """
    Process and display prediction results for each view.
    
    Args:
        predictions: List of prediction dictionaries from model inference
    """
    print("\nProcessing prediction results...")
    
    for i, pred in enumerate(predictions):
        print(f"\n--- View {i+1} ---")
        
        # Geometry outputs
        pts3d = pred["pts3d"]                     # 3D points in world coordinates (B, H, W, 3)
        pts3d_cam = pred["pts3d_cam"]             # 3D points in camera coordinates (B, H, W, 3)
        depth_z = pred["depth_z"]                 # Z-depth in camera frame (B, H, W, 1)
        depth_along_ray = pred["depth_along_ray"] # Depth along ray in camera frame (B, H, W, 1)

        # Camera outputs
        ray_directions = pred["ray_directions"]   # Ray directions in camera frame (B, H, W, 3)
        intrinsics = pred["intrinsics"]           # Recovered pinhole camera intrinsics (B, 3, 3)
        camera_poses = pred["camera_poses"]       # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world poses in world frame (B, 4, 4)
        cam_trans = pred["cam_trans"]             # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world translation in world frame (B, 3)
        cam_quats = pred["cam_quats"]             # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world quaternion in world frame (B, 4)

        # Quality and masking
        confidence = pred["conf"]                 # Per-pixel confidence scores (B, H, W)
        mask = pred["mask"]                       # Combined validity mask (B, H, W, 1)
        non_ambiguous_mask = pred["non_ambiguous_mask"]                # Non-ambiguous regions (B, H, W)
        non_ambiguous_mask_logits = pred["non_ambiguous_mask_logits"]  # Mask logits (B, H, W)

        # Scaling
        metric_scaling_factor = pred["metric_scaling_factor"]  # Applied metric scaling (B,)

        # Original input
        img_no_norm = pred["img_no_norm"]         # Denormalized input images for visualization (B, H, W, 3)
        
        # Print some basic info about the prediction
        print(f"  3D points shape: {pts3d.shape}")
        print(f"  Depth shape: {depth_z.shape}")
        depth_to_png(depth_z,path=f"outputs/depth_map_view_{i+1}.png")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pose-free evaluation using MapAnything model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--image_folder", 
        type=str, 
        required=True,
        help="Path to folder containing images to process"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/map-anything",
        help="Name of the MapAnything model to use. For Apache 2.0 license, use 'facebook/map-anything-apache'"
    )
    
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Use memory efficient inference (trades off speed for more views)"
    )
    
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable mixed precision inference"
    )
    
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Mixed precision data type"
    )
    
    parser.add_argument(
        "--no_mask",
        action="store_true",
        help="Disable masking to dense geometry outputs"
    )
    
    parser.add_argument(
        "--no_mask_edges",
        action="store_true",
        help="Disable edge artifact removal"
    )
    
    parser.add_argument(
        "--apply_confidence_mask",
        action="store_true",
        help="Filter low-confidence regions"
    )
    
    parser.add_argument(
        "--confidence_percentile",
        type=int,
        default=10,
        help="Remove bottom percentile confidence pixels"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the pose-free evaluation."""
    args = parse_arguments()
    
    try:
        # Setup device
        device = setup_device()
        
        # Load model
        model = load_model(device, args.model_name)
        
        # Load images
        views = load_images_from_folder(args.image_folder)
        
        # Run inference
        predictions = run_inference(
            model, 
            views,
            memory_efficient=args.memory_efficient,
            use_amp=not args.no_amp,
            amp_dtype=args.amp_dtype,
            apply_mask=not args.no_mask,
            mask_edges=not args.no_mask_edges,
            apply_confidence_mask=args.apply_confidence_mask,
            confidence_percentile=args.confidence_percentile
        )
        
        # Process results
        process_predictions(predictions)
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()