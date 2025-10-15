# Multi-Scale Performance Evaluation for MapAnything

This repository contains tools to evaluate how feedforward pose prediction models (like MapAnything) perform as the distance between images increases. The evaluation uses the aerial-megadepth dataset with COLMAP-calibrated poses as ground truth.

## 🎯 Overview

The evaluation pipeline:

1. **Loads COLMAP data**: Reads camera intrinsics and extrinsic poses from the aerial-megadepth dataset
2. **Selects image pairs**: Picks a base image and pairs it with other images at varying distances
3. **Predicts poses**: Runs MapAnything to predict relative poses between image pairs
4. **Compares results**: Calculates rotation and translation errors against COLMAP ground truth
5. **Visualizes trends**: Generates plots showing how accuracy degrades with increasing view distance

## 📁 Files

- **`multi_scale_eval.py`**: Main evaluation script
- **`list_scenes.py`**: Helper to list available scenes and their statistics
- **`pose_free_eval.py`**: Reference example for using MapAnything
- **`MULTI_SCALE_EVALUATION.md`**: Detailed documentation
- **`README_MULTI_SCALE.md`**: This file

## 🚀 Quick Start

### 1. List Available Scenes

First, see what scenes are available:

```bash
python list_scenes.py
```

For detailed information:

```bash
python list_scenes.py --details
```

### 2. Run Evaluation

Evaluate a scene with default settings (20 pairs):

```bash
python multi_scale_eval.py --scene 0000
```

### 3. View Results

Results are saved in `./multi_scale_results/`:
- `0000_results.csv`: Detailed metrics for each image pair
- `0000_distance_vs_accuracy.png`: Visualization of distance vs error

## 📊 Example Usage

### Basic Evaluation

```bash
# Evaluate scene 0000 with default settings
python multi_scale_eval.py --scene 0000
```

### Choose Specific Base Image

```bash
# Use a specific image as the reference
python multi_scale_eval.py --scene 0000 --base_image 0000_100.jpeg
```

### Evaluate More Pairs

```bash
# Test 50 image pairs instead of the default 20
python multi_scale_eval.py --scene 0000 --num_pairs 50
```

### Save to Custom Location

```bash
# Save results to a specific directory
python multi_scale_eval.py --scene 0000 --output_dir ./my_results
```

### Memory-Efficient Mode

```bash
# Use memory-efficient inference for limited GPU memory
python multi_scale_eval.py --scene 0000 --memory_efficient
```

### Complete Example

```bash
# Full example with all options
python multi_scale_eval.py \
    --scene 0001 \
    --base_image 0001_050.jpeg \
    --num_pairs 30 \
    --output_dir ./results_scene_0001 \
    --memory_efficient
```

## 📈 Understanding the Output

### Console Output

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
  Distance: 45.12m, Rot Error: 6.78°, Trans Error: 0.456m
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

### CSV Format

The CSV file contains:

| Column | Description |
|--------|-------------|
| `base_image` | Reference image name |
| `target_image` | Target image name |
| `distance` | 3D distance between camera centers (meters) |
| `rotation_error_deg` | Angular error in rotation (degrees) |
| `translation_error` | Euclidean error in translation (meters) |
| `baseline_length` | Distance between views (same as distance) |

### Visualization

The plot shows two subplots:

1. **Left**: Rotation Error (degrees) vs Distance (meters)
   - Shows how rotation prediction degrades with distance
   - Includes trend line

2. **Right**: Translation Error (meters) vs Distance (meters)
   - Shows how translation prediction degrades with distance
   - Includes trend line

**Expected Behavior**: Both errors typically increase with distance, as wider baselines make pose estimation more challenging.

## 🔧 Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_root` | str | `./multi-scale-dataset/aerial-megadepth` | Dataset root directory |
| `--scene` | str | **Required** | Scene to evaluate (e.g., '0000') |
| `--base_image` | str | None | Base image name (auto-select if not specified) |
| `--num_pairs` | int | 20 | Number of image pairs to evaluate |
| `--output_dir` | str | `./multi_scale_results` | Output directory |
| `--model_name` | str | `facebook/map-anything` | Model name |
| `--memory_efficient` | flag | False | Enable memory-efficient mode |

## 🗺️ Dataset Structure

The aerial-megadepth dataset should be organized as:

```
multi-scale-dataset/
└── aerial-megadepth/
    ├── 0000/
    │   ├── 0000.json
    │   └── sfm_output_localization/
    │       └── sfm_superpoint+superglue/
    │           └── localized_dense_metric/
    │               ├── images/            # Source images
    │               │   ├── 0000_000.jpeg
    │               │   ├── 0000_001.jpeg
    │               │   └── ...
    │               ├── sparse/            # Binary COLMAP files
    │               │   ├── cameras.bin
    │               │   ├── images.bin
    │               │   └── points3D.bin
    │               └── sparse-txt/        # Text COLMAP files (used by script)
    │                   ├── cameras.txt    # Camera intrinsics
    │                   ├── images.txt     # Camera poses
    │                   └── points3D.txt   # 3D points
    ├── 0001/
    ├── 0002/
    └── ...
```

## 📝 Available Scenes

The dataset contains **137 valid scenes** with over **132,000 images** total.

Common scenes include:
- Small scenes: 0000, 0001, 0002, 0003, 0004, 0005
- Medium scenes: 0011, 0012, 0013, 0015, 0016, 0017
- Large scenes: 0860, 1017, 1589

Use `python list_scenes.py --details` to see the number of images in each scene.

## 🎓 Methodology

### Coordinate Systems

- **COLMAP**: World-to-camera transformation
  - Rotation: Quaternion (w, x, y, z)
  - Translation: World-to-camera translation
  
- **MapAnything**: Cam-to-world transformation (OpenCV)
  - Format: 4×4 homogeneous transformation matrix
  - Convention: +X right, +Y down, +Z forward

The script automatically handles conversion between these conventions.

### Distance Metric

Camera distance = Euclidean distance between camera centers in world coordinates:

```
distance = ||C₁ - C₂||₂
```

### Error Metrics

1. **Rotation Error**:
   ```
   error = arccos((trace(R_error) - 1) / 2)
   ```
   - Represents the angle of the rotation error
   - Reported in degrees

2. **Translation Error**:
   ```
   error = ||t_pred - t_gt||₂
   ```
   - Euclidean distance between predicted and ground truth translations
   - Reported in meters

### Pair Selection Strategy

Image pairs are selected to provide even coverage across the distance range:

1. Compute distances from base image to all other images
2. Sort by distance
3. Select evenly spaced samples using `np.linspace`

This ensures evaluation spans the full range from close views to distant views.

## 💡 Tips and Best Practices

### Scene Selection

- **Start small**: Begin with scenes that have fewer images for faster iteration
- **Check coverage**: Use `list_scenes.py --details` to see image counts
- **Diverse baselines**: Try different base images to get varied perspectives

### Base Image Selection

- Choose images from well-reconstructed areas
- Avoid images at the edge of the reconstruction
- Select images with good overlap with other views

### Number of Pairs

- **Quick test**: `--num_pairs 10` for fast evaluation
- **Standard**: `--num_pairs 20` (default) for balanced coverage
- **Comprehensive**: `--num_pairs 50` for detailed analysis

### Memory Management

If you encounter GPU memory issues:
1. Use `--memory_efficient` flag
2. Reduce `--num_pairs`
3. Close other GPU processes
4. Monitor with `nvidia-smi`

### Interpreting Results

- **Low rotation error (<5°)**: Good alignment
- **Medium rotation error (5-15°)**: Acceptable for many applications
- **High rotation error (>15°)**: Poor alignment

- **Translation error**: Compare relative to baseline length
  - Normalized error = translation_error / baseline_length
  - Good: <5%
  - Acceptable: 5-10%
  - Poor: >10%

## 🔍 Troubleshooting

### Scene not found

```
Error: Scene not found: /path/to/scene
```

**Solutions**:
- Verify scene name with `python list_scenes.py`
- Check dataset root path with `--dataset_root`

### Image file not found

```
Error: Image file not found
```

**Solutions**:
- Verify images are in the `images/` subdirectory
- Check file extensions (`.jpeg` vs `.jpg`)
- Ensure image names match those in `images.txt`

### CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
- Use `--memory_efficient` flag
- Reduce `--num_pairs`
- Close other GPU applications
- Use smaller batch size in model

### Model download issues

```
Error loading model
```

**Solutions**:
- Ensure internet connection (first run downloads model)
- Check Hugging Face hub access
- Try alternative model: `--model_name facebook/map-anything-apache`

### No valid image pairs

```
Error: No valid image pairs found
```

**Solutions**:
- Check that images exist in the images directory
- Verify COLMAP files contain corresponding entries
- Try a different base image

## 📚 References

### MapAnything

```bibtex
@article{mapanything2024,
  title={MapAnything: Universal Visual Localization},
  author={...},
  journal={...},
  year={2024}
}
```

### COLMAP

```bibtex
@inproceedings{schoenberger2016sfm,
  author={Sch\"{o}nberger, Johannes L. and Frahm, Jan-Michael},
  title={Structure-from-Motion Revisited},
  booktitle={CVPR},
  year={2016}
}
```

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the detailed documentation in `MULTI_SCALE_EVALUATION.md`
3. Examine example output and plots

## 🔄 Example Workflow

Complete workflow for evaluating multiple scenes:

```bash
# 1. List available scenes
python list_scenes.py --details

# 2. Quick test on a small scene
python multi_scale_eval.py --scene 0000 --num_pairs 10

# 3. Full evaluation on multiple scenes
for scene in 0000 0001 0002 0003 0004; do
    echo "Evaluating scene $scene..."
    python multi_scale_eval.py \
        --scene $scene \
        --num_pairs 30 \
        --output_dir ./results_all_scenes \
        --memory_efficient
done

# 4. Compare results
ls -lh ./results_all_scenes/
```

## 📊 Batch Processing

To evaluate all scenes automatically:

```bash
#!/bin/bash
# batch_eval.sh

OUTPUT_DIR="./batch_results"
mkdir -p $OUTPUT_DIR

# Get list of scenes
python list_scenes.py | grep -E "^  [0-9]" | while read -r scenes; do
    for scene in $scenes; do
        echo "Processing scene: $scene"
        python multi_scale_eval.py \
            --scene $scene \
            --num_pairs 20 \
            --output_dir $OUTPUT_DIR \
            --memory_efficient
    done
done

echo "Batch processing complete!"
```

Make it executable and run:

```bash
chmod +x batch_eval.sh
./batch_eval.sh
```

## 🎯 Key Features

- ✅ Automatic COLMAP data parsing
- ✅ Coordinate system conversion handling
- ✅ Evenly distributed distance sampling
- ✅ Comprehensive error metrics
- ✅ Automatic visualization
- ✅ Memory-efficient mode
- ✅ Batch processing support
- ✅ Detailed logging and progress tracking
- ✅ Robust error handling
- ✅ CSV export for further analysis

## 📈 Future Enhancements

Potential improvements:
- [ ] Multi-GPU support
- [ ] Parallel image pair processing
- [ ] Additional error metrics (AUC, median error)
- [ ] Comparison with other methods
- [ ] Interactive visualization
- [ ] Automatic report generation
- [ ] Database caching for faster reruns

---

**Happy Evaluating! 🚀**

