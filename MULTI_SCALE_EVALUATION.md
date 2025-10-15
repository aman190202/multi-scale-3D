# Multi-Scale Evaluation for Feedforward Pose Prediction

This tool evaluates how feedforward models like MapAnything perform as the distance between images of the same scene increases.

## Overview

The evaluation script:
1. Loads COLMAP calibrated poses from the aerial-megadepth dataset
2. Selects a base image and pairs it with other images at increasing distances
3. Runs the MapAnything model to predict relative poses
4. Compares predicted poses with COLMAP ground truth
5. Generates CSV results and visualization plots

## Dataset Structure

The aerial-megadepth dataset should be organized as:
```
multi-scale-dataset/
└── aerial-megadepth/
    ├── 0000/
    │   ├── 0000.json
    │   └── sfm_output_localization/
    │       └── sfm_superpoint+superglue/
    │           └── localized_dense_metric/
    │               ├── images/
    │               │   ├── 0000_000.jpeg
    │               │   ├── 0000_001.jpeg
    │               │   └── ...
    │               └── sparse-txt/
    │                   ├── cameras.txt
    │                   ├── images.txt
    │                   └── points3D.txt
    ├── 0001/
    ├── 0002/
    └── ...
```

## Usage

### Basic Usage

Evaluate a scene with default settings:

```bash
python multi_scale_eval.py --scene 0000
```

### Specify Base Image

Choose a specific image as the base reference:

```bash
python multi_scale_eval.py --scene 0000 --base_image 0000_050.jpeg
```

### Adjust Number of Pairs

Evaluate more or fewer image pairs:

```bash
python multi_scale_eval.py --scene 0000 --num_pairs 50
```

### Full Example with All Options

```bash
python multi_scale_eval.py \
    --scene 0000 \
    --base_image 0000_100.jpeg \
    --num_pairs 30 \
    --output_dir ./results \
    --model_name facebook/map-anything \
    --memory_efficient
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_root` | str | `./multi-scale-dataset/aerial-megadepth` | Root directory of the dataset |
| `--scene` | str | **Required** | Scene name (e.g., '0000', '0001') |
| `--base_image` | str | None | Base image name (first image if not specified) |
| `--num_pairs` | int | 20 | Number of image pairs to evaluate |
| `--output_dir` | str | `./multi_scale_results` | Directory to save results |
| `--model_name` | str | `facebook/map-anything` | MapAnything model name |
| `--memory_efficient` | flag | False | Use memory efficient inference |

## Output

The script generates:

1. **CSV file** (`{scene}_results.csv`): Contains detailed results for each pair
   - Columns: `base_image`, `target_image`, `distance`, `rotation_error_deg`, `translation_error`, `baseline_length`

2. **Visualization plot** (`{scene}_distance_vs_accuracy.png`): Shows:
   - Left: Rotation error (degrees) vs distance
   - Right: Translation error (meters) vs distance
   - Includes trend lines to show how accuracy degrades with distance

3. **Console output**: Summary statistics including:
   - Number of pairs evaluated
   - Distance range
   - Mean and standard deviation of rotation and translation errors

## Example Output

```
Multi-Scale Evaluation
============================================================
Scene: 0000
Dataset root: /path/to/multi-scale-dataset/aerial-megadepth
Output directory: ./multi_scale_results
============================================================

Loaded 1289 cameras and 600 images
Selected base image: 0000_000.jpeg
Created 20 pairs with distances from 5.32 to 450.67

Evaluating 20 pairs...
  Distance: 5.32m, Rot Error: 2.45°, Trans Error: 0.125m
  Distance: 23.76m, Rot Error: 4.32°, Trans Error: 0.234m
  ...

Results saved to ./multi_scale_results/0000_results.csv
Plot saved to ./multi_scale_results/0000_distance_vs_accuracy.png

============================================================
Evaluation completed successfully!
============================================================

Summary Statistics:
  Number of pairs: 20
  Distance range: 5.32m to 450.67m
  Mean rotation error: 8.45° (±3.21°)
  Mean translation error: 0.456m (±0.234m)
```

## Available Scenes

To list all available scenes:

```bash
ls multi-scale-dataset/aerial-megadepth/
```

Common scenes include: 0000, 0001, 0002, 0003, 0004, 0005, 0007, 0008, etc.

## Methodology

### Pose Representation

- **COLMAP convention**: World-to-camera transformation with quaternion (w, x, y, z) and translation
- **MapAnything convention**: OpenCV cam-to-world transformation (4x4 matrix)
- The script handles conversion between these conventions

### Distance Metric

Camera distance is computed as the Euclidean distance between camera centers in world coordinates:
```
distance = ||C1 - C2||
```

### Pose Error Metrics

1. **Rotation Error**: Angular difference between predicted and ground truth rotations
   - Computed as: `arccos((trace(R_error) - 1) / 2)`
   - Reported in degrees

2. **Translation Error**: Euclidean distance between predicted and ground truth translations
   - Reported in meters

### Pair Selection

Image pairs are selected with evenly distributed distances to cover the full range from closest to farthest views.

## Tips

1. **Memory Management**: Use `--memory_efficient` flag if you encounter GPU memory issues
2. **Scene Selection**: Start with smaller scenes for faster evaluation
3. **Baseline Selection**: Choose a base image from a well-reconstructed area with good coverage
4. **Interpretation**: Look for trends in the plots - typically errors increase with distance

## Troubleshooting

### Scene not found
- Check that the scene directory exists
- Verify the dataset_root path is correct

### Image file not found
- Ensure images are in the `images/` subdirectory
- Check image file extensions match (`.jpeg` vs `.jpg`)

### CUDA out of memory
- Use `--memory_efficient` flag
- Reduce `--num_pairs`
- Close other GPU processes

### Model loading issues
- Ensure you have internet connection (first time downloads model)
- Check huggingface hub cache
- Try `facebook/map-anything-apache` for Apache 2.0 license version

## Citation

If you use this evaluation tool, please cite:

```bibtex
@article{mapanything2024,
  title={MapAnything: Universal Visual Localization},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This evaluation tool is provided as-is for research purposes.

