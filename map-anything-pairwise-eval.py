#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc.
#
# This script evaluates MapAnything pose predictions across increasing view gaps.
#
# For a fixed reference frame (default: frame_00001.jpeg) it samples target frames
# using a configurable stride, runs MapAnything on each (reference, target) pair,
# and compares the predicted relative pose against COLMAP ground truth stored in
# transforms.json. Rotation and translation-direction errors are saved to CSV so
# they can be plotted later.

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as tvf

from mapanything.models import MapAnything
from mapanything.utils.misc import seed_everything
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MapAnything relative pose errors versus COLMAP ground truth "
            "for (reference, target) image pairs."
        )
    )
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="Directory containing the scene images and transforms.json.",
    )
    parser.add_argument(
        "--transforms_path",
        type=str,
        default=None,
        help="Path to COLMAP-derived transforms.json (defaults to <scene_dir>/transforms.json).",
    )
    parser.add_argument(
        "--reference_image",
        type=str,
        default="frame_00001.jpeg",
        help="Image filename (within scene_dir/images) used as the anchor view.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Frame stride applied after the reference index when selecting target images.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Optional cap on the number of (reference, target) pairs to evaluate.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="pairwise_pose_errors.csv",
        help="Path to the CSV file where per-pair errors will be written.",
    )
    parser.add_argument(
        "--image_glob",
        type=str,
        default="*.jpeg",
        help="Glob pattern that defines which files inside <scene_dir>/images are considered.",
    )
    parser.add_argument(
        "--img_load_resolution",
        type=int,
        default=1024,
        help="Resolution used when loading images before feeding them to MapAnything.",
    )
    parser.add_argument(
        "--mapanything_resolution",
        type=int,
        default=518,
        help="Square resolution used as MapAnything input (keep at 518 unless you know what you're doing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for any stochastic MapAnything components.",
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        help="Use MapAnything memory-efficient inference (trades speed for lower VRAM).",
    )
    return parser.parse_args()


def load_and_preprocess_images_square(
    image_path_list: Sequence[str],
    target_size: int = 1024,
    data_norm_type: Optional[str] = None,
) -> torch.Tensor:
    """Loads, pads to square, resizes, and normalizes images."""
    if len(image_path_list) == 0:
        raise ValueError("At least one image is required")

    if data_norm_type is None:
        img_transform = tvf.ToTensor()
    elif data_norm_type in IMAGE_NORMALIZATION_DICT:
        img_norm = IMAGE_NORMALIZATION_DICT[data_norm_type]
        img_transform = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    else:
        raise ValueError(
            f"Unknown image normalization type: {data_norm_type}. "
            f"Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )

    processed_images: List[torch.Tensor] = []
    for image_path in image_path_list:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size
        max_dim = max(width, height)
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))
        square_img = square_img.resize((target_size, target_size), Image.BILINEAR)

        processed_images.append(img_transform(square_img))

    return torch.stack(processed_images, dim=0)


def run_mapanything_camera_inference(
    model: MapAnything,
    images: torch.Tensor,
    resolution: int = 518,
    image_normalization_type: str = "dinov2",
    memory_efficient_inference: bool = False,
) -> np.ndarray:
    """Runs MapAnything and returns the predicted camera-to-world matrices."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected images tensor of shape (N, 3, H, W)")

    resized = F.interpolate(
        images, size=(resolution, resolution), mode="bilinear", align_corners=False
    )

    views = []
    for view_idx in range(resized.shape[0]):
        view = {
            "img": resized[view_idx][None],
            "data_norm_type": [image_normalization_type],
        }
        views.append(view)

    with torch.no_grad():
        predictions = model.infer(
            views, memory_efficient_inference=memory_efficient_inference
        )

    camera_poses: List[np.ndarray] = []
    for pred in predictions:
        pose = pred["camera_poses"][0].cpu().numpy()
        camera_poses.append(pose)

    return np.stack(camera_poses, axis=0)


def load_ground_truth_c2w(transforms_path: Path) -> Dict[str, np.ndarray]:
    """Loads COLMAP-derived c2w matrices keyed by image filename."""
    with open(transforms_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pose_map: Dict[str, np.ndarray] = {}
    for frame in data.get("frames", []):
        file_name = Path(frame["file_path"]).name
        pose_map[file_name] = np.array(frame["transform_matrix"], dtype=np.float64)
    if not pose_map:
        raise ValueError(f"No frames found inside {transforms_path}")
    return pose_map


def build_target_paths(
    image_paths: Sequence[str],
    reference_index: int,
    stride: int,
    max_pairs: Optional[int],
) -> List[str]:
    if stride <= 0:
        raise ValueError("Stride must be positive")
    start_idx = reference_index
    targets = image_paths[start_idx + stride :: stride]
    if max_pairs is not None:
        targets = targets[:max_pairs]
    if len(targets) == 0:
        raise ValueError("No target frames selected; adjust stride/max_pairs")
    return targets


def relative_rotation_translation(c2w_i: np.ndarray, c2w_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns rotation (3x3) and translation (3,) from i -> j using c2w matrices."""
    rel = np.linalg.inv(c2w_j) @ c2w_i
    return rel[:3, :3], rel[:3, 3]


def geodesic_rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    cos_theta = (np.trace(R_pred @ R_gt.T) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def rotation_magnitude_degrees(R: np.ndarray) -> float:
    """Returns the geodesic angle between rotation R and identity."""
    return geodesic_rotation_error(R, np.eye(3))


def translation_direction_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    norm_pred = np.linalg.norm(t_pred)
    norm_gt = np.linalg.norm(t_gt)
    if norm_pred < 1e-8 or norm_gt < 1e-8:
        return float("nan")
    t_pred_hat = t_pred / norm_pred
    t_gt_hat = t_gt / norm_gt
    cos_theta = float(np.clip(np.dot(t_pred_hat, t_gt_hat), -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        major_capability = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major_capability >= 8 else torch.float16
    else:
        dtype = torch.float32

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    transforms_path = (
        Path(args.transforms_path)
        if args.transforms_path is not None
        else Path(args.scene_dir) / "transforms.json"
    )
    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found at {transforms_path}")
    gt_pose_map = load_ground_truth_c2w(transforms_path)

    image_dir = Path(args.scene_dir) / "images"
    image_paths = sorted(glob.glob(str(image_dir / args.image_glob)))
    if len(image_paths) == 0:
        raise ValueError(f"No images found under {image_dir} with pattern {args.image_glob}")

    reference_path = str(image_dir / args.reference_image)
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference image {reference_path} does not exist")

    if reference_path not in image_paths:
        raise ValueError(f"Reference image {reference_path} not found in image list")
    reference_index = image_paths.index(reference_path)
    target_paths = build_target_paths(image_paths, reference_index, args.stride, args.max_pairs)
    path_to_index = {path: idx for idx, path in enumerate(image_paths)}

    print(f"Loaded {len(image_paths)} candidate frames.")
    print(f"Evaluating {len(target_paths)} pairs with reference {Path(reference_path).name}.")

    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    model.eval()
    mapanything_norm = model.encoder.data_norm_type

    results = []
    for idx, target_path in enumerate(target_paths, start=1):
        pair_paths = [reference_path, target_path]
        print(f"[{idx}/{len(target_paths)}] Running MapAnything for {Path(target_path).name} ...")

        images = load_and_preprocess_images_square(
            pair_paths,
            target_size=args.img_load_resolution,
            data_norm_type=mapanything_norm,
        ).to(device=device)

        camera_poses = run_mapanything_camera_inference(
            model,
            images,
            resolution=args.mapanything_resolution,
            image_normalization_type=mapanything_norm,
            memory_efficient_inference=args.memory_efficient_inference,
        )

        ref_name = Path(reference_path).name
        tgt_name = Path(target_path).name

        pred_ref = camera_poses[0]
        pred_tgt = camera_poses[1]

        try:
            gt_ref = gt_pose_map[ref_name]
            gt_tgt = gt_pose_map[tgt_name]
        except KeyError as exc:
            raise KeyError(
                f"Missing ground-truth pose for {exc.args[0]} in {transforms_path}"
            ) from exc

        R_pred, t_pred = relative_rotation_translation(pred_ref, pred_tgt)
        R_gt, t_gt = relative_rotation_translation(gt_ref, gt_tgt)

        rot_err = geodesic_rotation_error(R_pred, R_gt)
        trans_err = translation_direction_error(t_pred, t_gt)
        frame_gap = path_to_index[target_path] - path_to_index[reference_path]
        gt_rot_mag = rotation_magnitude_degrees(R_gt)
        gt_translation_distance = float(np.linalg.norm(t_gt))

        print(
            f"Pair ({ref_name}, {tgt_name}) -> rotation error {rot_err:.3f} deg, "
            f"translation dir error {trans_err:.3f} deg"
        )

        results.append(
            {
                "reference_image": ref_name,
                "target_image": tgt_name,
                "frame_gap": frame_gap,
                "rotation_error_deg": rot_err,
                "translation_dir_error_deg": trans_err,
                "gt_rotation_deg": gt_rot_mag,
                "gt_translation_distance": gt_translation_distance,
            }
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "reference_image",
                "target_image",
                "frame_gap",
                "rotation_error_deg",
                "translation_dir_error_deg",
                "gt_rotation_deg",
                "gt_translation_distance",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Saved {len(results)} pairwise errors to {output_path}")


if __name__ == "__main__":
    main()
