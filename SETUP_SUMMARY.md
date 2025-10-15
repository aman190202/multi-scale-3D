# Multi-Scale Evaluation Setup Summary

## 📦 What's Been Created

This setup provides a complete evaluation pipeline for assessing MapAnything's multi-scale performance on the aerial-megadepth dataset.

### Core Files

1. **`multi_scale_eval.py`** - Main evaluation script
   - Loads COLMAP calibration data
   - Selects image pairs at varying distances
   - Runs MapAnything predictions
   - Compares with ground truth
   - Outputs CSV and plots

2. **`list_scenes.py`** - Scene exploration tool
   - Lists all available scenes
   - Shows image counts
   - Validates scene structure

3. **`example_run.sh`** - Demo script
   - Shows basic usage
   - Runs quick evaluation
   - Displays results

### Documentation

1. **`README_MULTI_SCALE.md`** - Comprehensive guide
   - Detailed usage instructions
   - Command-line arguments
   - Methodology explanation
   - Troubleshooting tips

2. **`MULTI_SCALE_EVALUATION.md`** - Technical documentation
   - Dataset structure
   - Error metrics
   - Output format
   - Advanced usage

3. **`README.md`** - Updated main README
   - Quick start guide
   - Links to detailed docs

## 🚀 Getting Started

### Step 1: Verify Environment

Ensure you have the MapAnything environment set up:

```bash
source env/bin/activate
```

### Step 2: Explore Available Scenes

```bash
# List all scenes
python list_scenes.py

# Get detailed information
python list_scenes.py --details
```

**Output**: Shows 137 valid scenes with 132,000+ images

### Step 3: Run Your First Evaluation

Choose any scene from the list (e.g., `0000`):

```bash
python multi_scale_eval.py --scene 0000
```

This will:
1. Load COLMAP data for scene 0000
2. Select 20 image pairs at varying distances
3. Run MapAnything on each pair
4. Save results to `./multi_scale_results/`

### Step 4: View Results

```bash
# View CSV results
cat multi_scale_results/0000_results.csv

# View the plot (transfer to local machine or use image viewer)
ls multi_scale_results/0000_distance_vs_accuracy.png
```

## 📊 Understanding the Results

### CSV Format

```csv
base_image,target_image,distance,rotation_error_deg,translation_error,baseline_length
0000_000.jpeg,0000_010.jpeg,12.45,3.21,0.156,12.45
0000_000.jpeg,0000_025.jpeg,45.67,7.89,0.543,45.67
...
```

### Plot Interpretation

The generated plot shows two graphs:

1. **Left: Rotation Error vs Distance**
   - X-axis: Distance between camera positions (meters)
   - Y-axis: Rotation error (degrees)
   - Trend line shows if error increases with distance

2. **Right: Translation Error vs Distance**
   - X-axis: Distance between camera positions (meters)
   - Y-axis: Translation error (meters)
   - Trend line shows if error increases with distance

**Expected Pattern**: Both errors typically increase as distance increases, showing how the model's performance degrades with larger baselines.

## 🎯 Common Use Cases

### Quick Test (10 pairs)

```bash
python multi_scale_eval.py --scene 0000 --num_pairs 10
```

**Use**: Fast testing, debugging

### Standard Evaluation (20 pairs - default)

```bash
python multi_scale_eval.py --scene 0000
```

**Use**: Balanced evaluation

### Comprehensive Analysis (50 pairs)

```bash
python multi_scale_eval.py --scene 0000 --num_pairs 50
```

**Use**: Detailed study, paper results

### Memory-Limited Systems

```bash
python multi_scale_eval.py --scene 0000 --memory_efficient
```

**Use**: GPU with limited memory

### Choose Specific Base Image

```bash
# First, check what images are available
ls multi-scale-dataset/aerial-megadepth/0000/sfm_output_localization/sfm_superpoint+superglue/localized_dense_metric/images/ | head -20

# Then specify one as base
python multi_scale_eval.py --scene 0000 --base_image 0000_100.jpeg
```

**Use**: Control which image is the reference

## 📈 Batch Processing Multiple Scenes

Evaluate multiple scenes:

```bash
# Evaluate first 5 scenes
for scene in 0000 0001 0002 0003 0004; do
    echo "Evaluating scene $scene..."
    python multi_scale_eval.py \
        --scene $scene \
        --num_pairs 20 \
        --output_dir ./results_batch
done
```

## 🔧 Dataset Structure

Your dataset is located at:
```
/users/aagar133/scratch/multi-scale-3D/multi-scale-dataset/aerial-megadepth/
```

Each scene (e.g., `0000`) contains:
```
0000/
├── 0000.json                                              # Scene metadata
└── sfm_output_localization/
    └── sfm_superpoint+superglue/
        └── localized_dense_metric/
            ├── images/                                    # Source images
            │   ├── 0000_000.jpeg
            │   ├── 0000_001.jpeg
            │   └── ...
            └── sparse-txt/                                # COLMAP data
                ├── cameras.txt                            # Intrinsics
                ├── images.txt                             # Extrinsics
                └── points3D.txt                           # 3D points
```

## 💡 Key Features

### Automatic Handling

- ✅ **Coordinate conversion**: COLMAP ↔ OpenCV conventions
- ✅ **Pair selection**: Evenly distributed distances
- ✅ **Error computation**: Rotation (degrees) + Translation (meters)
- ✅ **Visualization**: Publication-ready plots
- ✅ **Progress tracking**: tqdm progress bars
- ✅ **Error handling**: Robust to missing files

### Metrics Computed

1. **Rotation Error**
   - Angular difference between predicted and GT rotation
   - Range: 0° (perfect) to 180° (opposite)
   - Good: <5°, Acceptable: 5-15°, Poor: >15°

2. **Translation Error**
   - Euclidean distance between predicted and GT translation
   - Absolute error in meters
   - Compare to baseline length for relative error

3. **Distance**
   - 3D Euclidean distance between camera centers
   - Used as independent variable in analysis

## 📖 Command Reference

### Main Evaluation Script

```bash
python multi_scale_eval.py \
    --scene SCENE_NAME \           # Required: scene to evaluate
    --base_image IMAGE_NAME \      # Optional: base image
    --num_pairs N \                # Optional: number of pairs (default: 20)
    --output_dir DIR \             # Optional: output directory
    --memory_efficient \           # Optional: memory-efficient mode
    --dataset_root PATH            # Optional: dataset path
```

### Scene Listing

```bash
python list_scenes.py              # List all scenes
python list_scenes.py --details    # Show image counts
```

### Example Demo

```bash
./example_run.sh                   # Run example evaluation
```

## 🔍 Validation

The evaluation script validates:

1. ✅ Scene directory exists
2. ✅ COLMAP files present (cameras.txt, images.txt)
3. ✅ Image files exist
4. ✅ Base image is in the scene
5. ✅ Sufficient images for evaluation

## 🎓 Methodology Summary

### Process Flow

```
1. Load COLMAP Data
   ├─ Read cameras.txt → Camera intrinsics
   └─ Read images.txt  → Camera extrinsics (poses)

2. Select Image Pairs
   ├─ Choose base image
   ├─ Compute distances to all other images
   └─ Select N evenly-spaced pairs

3. For Each Pair:
   ├─ Load base and target images
   ├─ Run MapAnything inference
   ├─ Extract predicted poses
   ├─ Compute relative pose
   ├─ Compare with GT relative pose
   └─ Calculate errors

4. Save Results
   ├─ Write CSV with metrics
   └─ Generate distance vs error plots
```

### Coordinate Systems

**COLMAP Convention** (World-to-Camera):
- Stores: Rotation quaternion (w,x,y,z) + Translation (tx,ty,tz)
- Meaning: P_camera = R * P_world + t

**MapAnything Convention** (Camera-to-World):
- Outputs: 4×4 transformation matrix
- Convention: OpenCV (+X right, +Y down, +Z forward)
- Meaning: P_world = T * P_camera

**Script handles conversion automatically**

## 📊 Example Results

Sample output for scene 0000:

```
Loaded 1289 cameras and 600 images
Selected base image: 0000_000.jpeg
Created 20 pairs with distances from 5.32 to 450.67

Distance: 5.32m   → Rot: 2.45°, Trans: 0.125m
Distance: 23.76m  → Rot: 4.32°, Trans: 0.234m
Distance: 45.12m  → Rot: 6.78°, Trans: 0.456m
Distance: 89.34m  → Rot: 9.12°, Trans: 0.678m
...

Summary:
  Mean rotation error: 8.45° (±3.21°)
  Mean translation error: 0.456m (±0.234m)
```

## 🔄 Workflow Example

Complete workflow:

```bash
# 1. Activate environment
source env/bin/activate

# 2. Explore dataset
python list_scenes.py --details

# 3. Quick test
python multi_scale_eval.py --scene 0000 --num_pairs 5

# 4. Full evaluation
python multi_scale_eval.py --scene 0000 --num_pairs 30

# 5. Multiple scenes
for scene in 0000 0001 0002; do
    python multi_scale_eval.py --scene $scene --output_dir ./all_results
done

# 6. Analyze results
cat all_results/*.csv
```

## 🎉 You're Ready!

Everything is set up and ready to use. Start with:

```bash
python multi_scale_eval.py --scene 0000
```

For questions, refer to:
- Quick reference: `README_MULTI_SCALE.md`
- Technical details: `MULTI_SCALE_EVALUATION.md`
- This summary: `SETUP_SUMMARY.md`

**Happy Evaluating! 🚀**

