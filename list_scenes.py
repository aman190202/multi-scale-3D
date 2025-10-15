#!/usr/bin/env python3
"""
Helper script to list all available scenes in the aerial-megadepth dataset
and show basic information about each scene.
"""

import argparse
from pathlib import Path
import sys


def count_images(scene_path: Path) -> int:
    """Count number of images in a scene."""
    images_dir = scene_path / "sfm_output_localization" / "sfm_superpoint+superglue" / "localized_dense_metric" / "images"
    if images_dir.exists():
        return len(list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.jpg")))
    return 0


def check_scene_valid(scene_path: Path) -> bool:
    """Check if a scene has all required files."""
    sparse_txt_dir = scene_path / "sfm_output_localization" / "sfm_superpoint+superglue" / "localized_dense_metric" / "sparse-txt"
    
    required_files = ['cameras.txt', 'images.txt', 'points3D.txt']
    
    for file in required_files:
        if not (sparse_txt_dir / file).exists():
            return False
    
    return True


def list_scenes(dataset_root: Path, show_details: bool = False):
    """List all available scenes in the dataset."""
    
    if not dataset_root.exists():
        print(f"Error: Dataset root not found: {dataset_root}")
        sys.exit(1)
    
    # Get all scene directories
    scene_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    
    print(f"\n{'='*80}")
    print(f"Available Scenes in {dataset_root.name}")
    print(f"{'='*80}\n")
    
    valid_scenes = []
    invalid_scenes = []
    
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        is_valid = check_scene_valid(scene_dir)
        num_images = count_images(scene_dir) if is_valid else 0
        
        if is_valid:
            valid_scenes.append((scene_name, num_images))
        else:
            invalid_scenes.append(scene_name)
    
    # Print valid scenes
    if valid_scenes:
        print(f"Valid scenes ({len(valid_scenes)}):")
        print(f"{'-'*80}")
        
        if show_details:
            print(f"{'Scene':<15} {'Images':<10} {'Status':<20}")
            print(f"{'-'*80}")
            for scene_name, num_images in valid_scenes:
                print(f"{scene_name:<15} {num_images:<10} ✓ Ready")
        else:
            # Print in columns
            scenes_per_row = 8
            for i in range(0, len(valid_scenes), scenes_per_row):
                row_scenes = [s[0] for s in valid_scenes[i:i+scenes_per_row]]
                print("  " + "  ".join(f"{s:<8}" for s in row_scenes))
        
        print(f"\nTotal valid scenes: {len(valid_scenes)}")
        print(f"Total images: {sum(n for _, n in valid_scenes)}")
    else:
        print("No valid scenes found!")
    
    # Print invalid scenes if any
    if invalid_scenes:
        print(f"\n{'-'*80}")
        print(f"Invalid/Incomplete scenes ({len(invalid_scenes)}):")
        print(f"{'-'*80}")
        for scene_name in invalid_scenes:
            print(f"  {scene_name} - Missing required files")
    
    print(f"\n{'='*80}\n")
    
    # Print usage hint
    print("Usage example:")
    if valid_scenes:
        example_scene = valid_scenes[0][0]
        print(f"  python multi_scale_eval.py --scene {example_scene}")
        print(f"  python multi_scale_eval.py --scene {example_scene} --num_pairs 30\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="List available scenes in the aerial-megadepth dataset"
    )
    
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/users/aagar133/scratch/multi-scale-3D/multi-scale-dataset/aerial-megadepth",
        help="Root directory of the aerial megadepth dataset"
    )
    
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed information for each scene"
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset_root = Path(args.dataset_root)
    list_scenes(dataset_root, show_details=args.details)


if __name__ == "__main__":
    main()

