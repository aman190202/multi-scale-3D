# Quick Reference Card

## 🚀 Commands

### List Scenes
```bash
python list_scenes.py                    # Brief list
python list_scenes.py --details          # With image counts
```

### Run Evaluation
```bash
# Basic (default: 20 pairs)
python multi_scale_eval.py --scene 0000

# Quick test (10 pairs)
python multi_scale_eval.py --scene 0000 --num_pairs 10

# Comprehensive (50 pairs)
python multi_scale_eval.py --scene 0000 --num_pairs 50

# With specific base image
python multi_scale_eval.py --scene 0000 --base_image 0000_100.jpeg

# Memory efficient mode
python multi_scale_eval.py --scene 0000 --memory_efficient

# Custom output directory
python multi_scale_eval.py --scene 0000 --output_dir ./my_results
```

### Batch Processing
```bash
# Multiple scenes
for scene in 0000 0001 0002; do
    python multi_scale_eval.py --scene $scene --output_dir ./batch_results
done
```

## 📁 Output Files

```
./multi_scale_results/
├── 0000_results.csv                     # Detailed metrics
└── 0000_distance_vs_accuracy.png        # Visualization
```

## 📊 CSV Columns

| Column | Description |
|--------|-------------|
| `base_image` | Reference image |
| `target_image` | Target image |
| `distance` | Camera distance (m) |
| `rotation_error_deg` | Rotation error (°) |
| `translation_error` | Translation error (m) |
| `baseline_length` | Same as distance |

## 📈 Interpreting Results

### Rotation Error
- **Good**: < 5°
- **Acceptable**: 5-15°
- **Poor**: > 15°

### Translation Error
Compute relative error:
```
relative_error = translation_error / distance
```
- **Good**: < 5%
- **Acceptable**: 5-10%
- **Poor**: > 10%

## 🎯 Common Scenarios

### Debugging
```bash
python multi_scale_eval.py --scene 0000 --num_pairs 5
```

### Paper Results
```bash
python multi_scale_eval.py --scene 0000 --num_pairs 50
```

### Low Memory GPU
```bash
python multi_scale_eval.py --scene 0000 --memory_efficient
```

### Specific Image as Base
```bash
# List images
ls multi-scale-dataset/aerial-megadepth/0000/sfm_output_localization/sfm_superpoint+superglue/localized_dense_metric/images/

# Use one as base
python multi_scale_eval.py --scene 0000 --base_image 0000_050.jpeg
```

## 🔧 Troubleshooting

### Scene not found
```bash
python list_scenes.py  # Check valid scenes
```

### Out of memory
```bash
python multi_scale_eval.py --scene 0000 --memory_efficient
```

### No images found
```bash
# Check scene structure
ls multi-scale-dataset/aerial-megadepth/0000/sfm_output_localization/sfm_superpoint+superglue/localized_dense_metric/images/
```

## 📚 Documentation

- **Complete Guide**: `README_MULTI_SCALE.md`
- **Setup Summary**: `SETUP_SUMMARY.md`
- **Technical Details**: `MULTI_SCALE_EVALUATION.md`
- **This Card**: `QUICK_REFERENCE.md`

## 🎓 Key Concepts

**Distance**: Euclidean distance between camera centers

**Rotation Error**: Angular difference (degrees)

**Translation Error**: Euclidean distance (meters)

**Multi-Scale**: How accuracy changes with increasing view distance

## 📦 Dataset Info

- **Location**: `/users/aagar133/scratch/multi-scale-3D/multi-scale-dataset/aerial-megadepth/`
- **Scenes**: 137 valid scenes
- **Images**: 132,000+ total
- **Format**: JPEG images with COLMAP calibration

## 🔄 Typical Workflow

```bash
# 1. Explore
python list_scenes.py

# 2. Quick test
python multi_scale_eval.py --scene 0000 --num_pairs 10

# 3. Full run
python multi_scale_eval.py --scene 0000

# 4. View results
cat multi_scale_results/0000_results.csv
```

## 💡 Pro Tips

1. Start with fewer pairs for testing
2. Use `--memory_efficient` for large scenes
3. Choose base images from well-reconstructed areas
4. Check plots for trends: errors should increase with distance
5. Compare multiple scenes for robust conclusions

## 🎯 All Arguments

```bash
python multi_scale_eval.py \
    --scene SCENE_NAME \              # Required
    --base_image IMAGE_NAME \         # Optional
    --num_pairs N \                   # Default: 20
    --output_dir DIR \                # Default: ./multi_scale_results
    --memory_efficient \              # Flag
    --dataset_root PATH \             # Default: auto-detected
    --model_name MODEL                # Default: facebook/map-anything
```

## 📞 Help

```bash
python multi_scale_eval.py --help
python list_scenes.py --help
```

