#!/usr/bin/env python3
"""
Multi-scale evaluation script for feedforward models like MapAnything.
Evaluates how pose prediction accuracy changes with increasing distance between views.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import csv
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

# Optional config for better memory efficiency
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from PIL import Image


def read_colmap_cameras(cameras_file: Path) -> Dict:
    """
    Read COLMAP cameras.txt file and return camera parameters.
    
    Format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    """
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            
            cameras[camera_id] = {
                'id': camera_id,
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    
    return cameras


def read_colmap_images(images_file: Path) -> Dict:
    """
    Read COLMAP images.txt file and return image poses.
    
    Format (alternating lines):
    IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    POINTS2D[] as x, y, POINT3D_ID
    """
    images = {}
    with open(images_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        
        # Parse image line
        parts = line.split()
        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = ' '.join(parts[9:])  # Handle names with spaces
        
        # Store image info (skip points2d line)
        images[name] = {
            'id': image_id,
            'quat': np.array([qw, qx, qy, qz]),  # Quaternion (w, x, y, z)
            'tvec': np.array([tx, ty, tz]),  # Translation vector
            'camera_id': camera_id,
            'name': name
        }
        
        i += 2  # Skip the points2d line
    
    return images


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    """
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])
    
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [w, x, y, z].
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def get_camera_pose_matrix(quat: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Get 4x4 camera pose matrix from quaternion and translation.
    COLMAP uses world-to-camera convention: P_cam = R * P_world + t
    We convert to cam-to-world (camera pose in world frame).
    """
    R_w2c = quaternion_to_rotation_matrix(quat)
    t_w2c = tvec
    
    # Convert to cam2world
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c
    
    # Create 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R_c2w
    T[:3, 3] = t_c2w
    
    return T


def compute_camera_distance(pose1: np.ndarray, pose2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two camera poses.
    """
    return np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])


def compute_relative_pose(pose_base: np.ndarray, pose_target: np.ndarray) -> np.ndarray:
    """
    Compute relative pose from base to target.
    T_rel = T_base^-1 * T_target
    """
    T_base_inv = np.linalg.inv(pose_base)
    T_rel = T_base_inv @ pose_target
    return T_rel


def compute_pose_error(T_pred: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float]:
    """
    Compute rotation and translation errors between predicted and ground truth poses.
    
    Returns:
        rotation_error (degrees), translation_error (as fraction of baseline)
    """
    # Compute relative error
    T_error = np.linalg.inv(T_pred) @ T_gt
    
    # Rotation error (angle of rotation matrix)
    R_error = T_error[:3, :3]
    trace = np.trace(R_error)
    # Clamp trace to valid range [-1, 3] to avoid numerical issues
    trace = np.clip(trace, -1.0, 3.0)
    rotation_error = np.arccos((trace - 1) / 2) * 180 / np.pi
    
    # Translation error (Euclidean distance)
    t_error = np.linalg.norm(T_error[:3, 3])
    
    return rotation_error, t_error


def opencv_to_colmap_pose(T_opencv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert OpenCV cam2world pose to COLMAP world2cam quaternion and translation.
    OpenCV: +X right, +Y down, +Z forward
    COLMAP stores world-to-camera transformation
    """
    # T_opencv is cam2world, we need world2cam
    T_w2c = np.linalg.inv(T_opencv)
    
    R_w2c = T_w2c[:3, :3]
    t_w2c = T_w2c[:3, 3]
    
    quat = rotation_matrix_to_quaternion(R_w2c)
    
    return quat, t_w2c


def load_scene_data(scene_path: Path) -> Tuple[Dict, Dict, Path]:
    """
    Load COLMAP data and images for a scene.
    """
    # Locate sparse-txt directory
    sparse_txt_dir = scene_path / "sfm_output_localization" / "sfm_superpoint+superglue" / "localized_dense_metric" / "sparse-txt"
    images_dir = scene_path / "sfm_output_localization" / "sfm_superpoint+superglue" / "localized_dense_metric" / "images"
    
    if not sparse_txt_dir.exists():
        raise FileNotFoundError(f"sparse-txt directory not found at {sparse_txt_dir}")
    
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found at {images_dir}")
    
    # Read COLMAP files
    cameras = read_colmap_cameras(sparse_txt_dir / "cameras.txt")
    images = read_colmap_images(sparse_txt_dir / "images.txt")
    
    print(f"Loaded {len(cameras)} cameras and {len(images)} images")
    
    return cameras, images, images_dir


def select_image_pairs(images: Dict, images_dir: Path, base_image_name: str = None, 
                       num_pairs: int = 20) -> List[Tuple[str, str, float]]:
    """
    Select base image and create pairs with increasing distance.
    
    Returns:
        List of (base_image_name, target_image_name, distance) tuples sorted by distance
    """
    # Select base image
    if base_image_name is None:
        # Select the first available image as base
        base_image_name = list(images.keys())[0]
    
    if base_image_name not in images:
        # Try to find it with different extension or path
        found = False
        for img_name in images.keys():
            if base_image_name in img_name:
                base_image_name = img_name
                found = True
                break
        if not found:
            raise ValueError(f"Base image {base_image_name} not found in images")
    
    base_image = images[base_image_name]
    base_pose = get_camera_pose_matrix(base_image['quat'], base_image['tvec'])
    
    # Compute distances to all other images
    pairs = []
    for img_name, img_data in images.items():
        if img_name == base_image_name:
            continue
        
        # Check if image file exists
        img_path = images_dir / img_name
        if not img_path.exists():
            continue
        
        target_pose = get_camera_pose_matrix(img_data['quat'], img_data['tvec'])
        distance = compute_camera_distance(base_pose, target_pose)
        
        pairs.append((base_image_name, img_name, distance))
    
    # Sort by distance
    pairs.sort(key=lambda x: x[2])
    
    # Select evenly distributed pairs
    if len(pairs) > num_pairs:
        indices = np.linspace(0, len(pairs) - 1, num_pairs, dtype=int)
        pairs = [pairs[i] for i in indices]
    
    print(f"Selected base image: {base_image_name}")
    print(f"Created {len(pairs)} pairs with distances from {pairs[0][2]:.2f} to {pairs[-1][2]:.2f}")
    
    return pairs


def run_mapanything_on_pair(model, base_image_path: Path, target_image_path: Path, 
                             device: str, memory_efficient: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run MapAnything model on an image pair and extract relative pose.
    
    Returns:
        base_pose, target_pose (both 4x4 matrices in world frame)
    """
    # Create temporary directory for image pair
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Copy images to temp directory
        import shutil
        shutil.copy(base_image_path, tmp_path / base_image_path.name)
        shutil.copy(target_image_path, tmp_path / target_image_path.name)
        
        # Load images
        views = load_images(str(tmp_path))
        
        # Run inference
        predictions = model.infer(
            views,
            memory_efficient_inference=memory_efficient,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=False,
            mask_edges=False,
            apply_confidence_mask=False,
        )
    
    # Extract camera poses (OpenCV convention: cam2world)
    # predictions[0] is for the first image, predictions[1] for the second
    base_pose = predictions[0]["camera_poses"][0].cpu().numpy()  # (4, 4)
    target_pose = predictions[1]["camera_poses"][0].cpu().numpy()  # (4, 4)
    
    return base_pose, target_pose


def evaluate_scene(scene_path: Path, model, device: str, output_dir: Path, 
                    base_image_name: str = None, num_pairs: int = 20,
                    memory_efficient: bool = False):
    """
    Evaluate pose prediction on a single scene.
    """
    # Load scene data
    cameras, images, images_dir = load_scene_data(scene_path)
    
    # Select image pairs
    pairs = select_image_pairs(images, images_dir, base_image_name, num_pairs)
    
    # Prepare output CSV
    scene_name = scene_path.name
    csv_path = output_dir / f"{scene_name}_results.csv"
    
    results = []
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['base_image', 'target_image', 'distance', 'rotation_error_deg', 
                        'translation_error', 'baseline_length'])
        
        print(f"\nEvaluating {len(pairs)} pairs...")
        for base_name, target_name, distance in tqdm(pairs):
            try:
                # Get ground truth poses
                base_gt = images[base_name]
                target_gt = images[target_name]
                
                base_pose_gt = get_camera_pose_matrix(base_gt['quat'], base_gt['tvec'])
                target_pose_gt = get_camera_pose_matrix(target_gt['quat'], target_gt['tvec'])
                
                # Compute ground truth relative pose
                T_rel_gt = compute_relative_pose(base_pose_gt, target_pose_gt)
                
                # Run model prediction
                base_image_path = images_dir / base_name
                target_image_path = images_dir / target_name
                
                base_pose_pred, target_pose_pred = run_mapanything_on_pair(
                    model, base_image_path, target_image_path, device, memory_efficient
                )
                
                # Compute predicted relative pose
                T_rel_pred = compute_relative_pose(base_pose_pred, target_pose_pred)
                
                # Compute errors
                rotation_error, translation_error = compute_pose_error(T_rel_pred, T_rel_gt)
                
                # Compute baseline length for normalization
                baseline_length = distance
                
                # Store results
                writer.writerow([base_name, target_name, distance, rotation_error, 
                               translation_error, baseline_length])
                
                results.append({
                    'distance': distance,
                    'rotation_error': rotation_error,
                    'translation_error': translation_error,
                    'baseline': baseline_length
                })
                
                print(f"  Distance: {distance:.2f}m, Rot Error: {rotation_error:.2f}°, "
                      f"Trans Error: {translation_error:.3f}m")
                
            except Exception as e:
                print(f"  Error processing pair ({base_name}, {target_name}): {e}")
                continue
    
    print(f"\nResults saved to {csv_path}")
    
    # Plot results
    plot_results(results, output_dir, scene_name)
    
    return results


def plot_results(results: List[Dict], output_dir: Path, scene_name: str):
    """
    Plot distance vs pose accuracy.
    """
    if not results:
        print("No results to plot")
        return
    
    distances = [r['distance'] for r in results]
    rotation_errors = [r['rotation_error'] for r in results]
    translation_errors = [r['translation_error'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot rotation error
    ax1.scatter(distances, rotation_errors, alpha=0.6)
    ax1.set_xlabel('Distance between views (m)', fontsize=12)
    ax1.set_ylabel('Rotation Error (degrees)', fontsize=12)
    ax1.set_title(f'Rotation Error vs Distance\nScene: {scene_name}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(distances, rotation_errors, 1)
    p = np.poly1d(z)
    ax1.plot(distances, p(distances), "r--", alpha=0.8, label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')
    ax1.legend()
    
    # Plot translation error
    ax2.scatter(distances, translation_errors, alpha=0.6, color='orange')
    ax2.set_xlabel('Distance between views (m)', fontsize=12)
    ax2.set_ylabel('Translation Error (m)', fontsize=12)
    ax2.set_title(f'Translation Error vs Distance\nScene: {scene_name}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(distances, translation_errors, 1)
    p = np.poly1d(z)
    ax2.plot(distances, p(distances), "r--", alpha=0.8, label=f'Trend: y={z[0]:.4f}x+{z[1]:.3f}')
    ax2.legend()
    
    plt.tight_layout()
    
    plot_path = output_dir / f"{scene_name}_distance_vs_accuracy.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    
    plt.close()


def setup_device():
    """Setup and return the appropriate device for inference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def load_model(device, model_name="facebook/map-anything"):
    """Load the MapAnything model."""
    print(f"Loading model: {model_name}")
    model = MapAnything.from_pretrained(model_name).to(device)
    print("Model loaded successfully")
    return model


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-scale evaluation for feedforward pose prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/users/aagar133/scratch/multi-scale-3D/multi-scale-dataset/aerial-megadepth",
        help="Root directory of the aerial megadepth dataset"
    )
    
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene name to evaluate (e.g., '0000', '0001', etc.)"
    )
    
    parser.add_argument(
        "--base_image",
        type=str,
        default=None,
        help="Name of the base image (if not specified, first image will be used)"
    )
    
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=20,
        help="Number of image pairs to evaluate"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./multi_scale_results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/map-anything",
        help="Name of the MapAnything model to use"
    )
    
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Use memory efficient inference"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the multi-scale evaluation."""
    args = parse_arguments()
    
    try:
        # Setup
        device = setup_device()
        model = load_model(device, args.model_name)
        
        # Prepare paths
        dataset_root = Path(args.dataset_root)
        scene_path = dataset_root / args.scene
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene not found: {scene_path}")
        
        print(f"\n{'='*60}")
        print(f"Multi-Scale Evaluation")
        print(f"{'='*60}")
        print(f"Scene: {args.scene}")
        print(f"Dataset root: {dataset_root}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")
        
        # Run evaluation
        results = evaluate_scene(
            scene_path=scene_path,
            model=model,
            device=device,
            output_dir=output_dir,
            base_image_name=args.base_image,
            num_pairs=args.num_pairs,
            memory_efficient=args.memory_efficient
        )
        
        print(f"\n{'='*60}")
        print("Evaluation completed successfully!")
        print(f"{'='*60}")
        
        # Print summary statistics
        if results:
            distances = [r['distance'] for r in results]
            rotation_errors = [r['rotation_error'] for r in results]
            translation_errors = [r['translation_error'] for r in results]
            
            print(f"\nSummary Statistics:")
            print(f"  Number of pairs: {len(results)}")
            print(f"  Distance range: {min(distances):.2f}m to {max(distances):.2f}m")
            print(f"  Mean rotation error: {np.mean(rotation_errors):.2f}° (±{np.std(rotation_errors):.2f}°)")
            print(f"  Mean translation error: {np.mean(translation_errors):.3f}m (±{np.std(translation_errors):.3f}m)")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

