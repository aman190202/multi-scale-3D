# 🎯 START HERE - Multi-Scale Evaluation System

## What Has Been Created

I've built a complete evaluation system to assess how MapAnything's pose prediction performance changes as the distance between images increases. This addresses your requirement to evaluate multi-scale performance on the aerial-megadepth dataset.

## 📦 Files Created

### Core Scripts
1. **`multi_scale_eval.py`** (20KB) - Main evaluation engine
2. **`list_scenes.py`** (4KB) - Scene exploration tool  
3. **`example_run.sh`** (2KB) - Demo script

### Documentation
4. **`README_MULTI_SCALE.md`** (13KB) - Complete user guide
5. **`SETUP_SUMMARY.md`** (9KB) - Setup and methodology
6. **`MULTI_SCALE_EVALUATION.md`** (6KB) - Technical details
7. **`QUICK_REFERENCE.md`** (4KB) - Command reference card
8. **`START_HERE.md`** - This file

### Updated Files
9. **`README.md`** - Updated with quick start info

## 🚀 Quick Start (30 seconds)

### Step 1: List available scenes
```bash
python list_scenes.py
```

**Expected output**: List of 137 scenes (0000, 0001, 0002, ...)

### Step 2: Run evaluation on a scene
```bash
python multi_scale_eval.py --scene 0000
```

**What it does**:
- Loads COLMAP poses from the aerial-megadepth dataset
- Selects 20 image pairs at increasing distances
- Runs MapAnything to predict relative poses
- Compares predictions with COLMAP ground truth
- Saves results to CSV and generates plots

**Time**: ~5-10 minutes per scene (depends on GPU)

### Step 3: View results
```bash
# View CSV
cat multi_scale_results/0000_results.csv

# View plot (transfer to local machine)
ls multi_scale_results/0000_distance_vs_accuracy.png
```

## 📊 What You Get

### 1. CSV File with Metrics
```csv
base_image,target_image,distance,rotation_error_deg,translation_error,baseline_length
0000_000.jpeg,0000_005.jpeg,12.45,3.21,0.156,12.45
0000_000.jpeg,0000_015.jpeg,34.67,5.43,0.289,34.67
0000_000.jpeg,0000_030.jpeg,78.90,8.76,0.512,78.90
...
```

### 2. Visualization Plot
Two subplots showing:
- **Left**: Rotation error (degrees) vs Distance (meters)
- **Right**: Translation error (meters) vs Distance (meters)

Both include trend lines to show how accuracy degrades with distance.

### 3. Summary Statistics
```
Summary Statistics:
  Number of pairs: 20
  Distance range: 5.32m to 450.67m
  Mean rotation error: 8.45° (±3.21°)
  Mean translation error: 0.456m (±0.234m)
```

## 🎓 How It Works

```
┌─────────────────────────────────────────────────────────┐
│ 1. LOAD COLMAP DATA                                     │
│    • Read cameras.txt (intrinsics)                      │
│    • Read images.txt (extrinsics/poses)                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 2. SELECT IMAGE PAIRS                                   │
│    • Pick base image                                    │
│    • Compute distances to all other images             │
│    • Select N pairs evenly distributed by distance     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 3. FOR EACH PAIR:                                       │
│    • Load base + target images                          │
│    • Run MapAnything inference                          │
│    • Extract predicted poses                            │
│    • Compute relative pose: T_rel = T_base⁻¹ × T_target│
│    • Compare with ground truth relative pose            │
│    • Calculate rotation & translation errors            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 4. OUTPUT RESULTS                                       │
│    • Save CSV with all metrics                          │
│    • Generate distance vs accuracy plots                │
│    • Print summary statistics                           │
└─────────────────────────────────────────────────────────┘
```

## 🎯 Common Use Cases

### Quick Test (5 pairs, fast)
```bash
python multi_scale_eval.py --scene 0000 --num_pairs 5
```

### Standard Evaluation (20 pairs, default)
```bash
python multi_scale_eval.py --scene 0000
```

### Comprehensive Study (50 pairs, thorough)
```bash
python multi_scale_eval.py --scene 0000 --num_pairs 50
```

### Multiple Scenes (batch processing)
```bash
for scene in 0000 0001 0002 0003 0004; do
    python multi_scale_eval.py --scene $scene --output_dir ./batch_results
done
```

### Low Memory GPU
```bash
python multi_scale_eval.py --scene 0000 --memory_efficient
```

## 📖 Documentation Guide

Choose your reading level:

- **5-minute read**: `QUICK_REFERENCE.md` - Command cheat sheet
- **15-minute read**: `SETUP_SUMMARY.md` - Setup and examples
- **30-minute read**: `README_MULTI_SCALE.md` - Complete guide
- **Technical deep-dive**: `MULTI_SCALE_EVALUATION.md` - Methodology

## 🎨 Key Features

✅ **Scene Selection**: Choose from 137 scenes with 132K+ images
✅ **Automatic Pair Selection**: Evenly distributed distances
✅ **Coordinate Conversion**: Handles COLMAP ↔ OpenCV conventions
✅ **Robust Error Metrics**: Rotation (degrees) + Translation (meters)
✅ **Publication-Ready Plots**: PNG with trend lines
✅ **CSV Export**: For further analysis in Excel/Python
✅ **Progress Tracking**: Real-time progress bars
✅ **Memory Efficient**: Optional mode for limited GPUs

## 📊 Dataset Overview

- **Location**: `multi-scale-dataset/aerial-megadepth/`
- **Scenes**: 137 valid scenes
- **Total Images**: 132,137 images
- **Image Types**: Mobile phone photos + Google Earth renders
- **Calibration**: COLMAP (Structure from Motion)
- **Format**: JPEG images with camera poses

## 🔍 What Makes This Multi-Scale?

The evaluation varies the **distance** between image pairs:

```
Pair 1:  Base ←─5m──→ Close View
Pair 2:  Base ←─20m─→ Medium View  
Pair 3:  Base ←─50m─→ Far View
Pair 4:  Base ←─100m→ Very Far View
...
```

This tests how well the model handles:
- **Small baselines**: Close views with large overlap
- **Medium baselines**: Moderate overlap
- **Large baselines**: Distant views with limited overlap

**Expected**: Errors increase with distance (wider baselines are harder)

## 💡 Understanding Results

### Good Performance
```
Distance: 10m  → Rotation: 2°,  Translation: 0.1m
Distance: 50m  → Rotation: 5°,  Translation: 0.3m
Distance: 100m → Rotation: 8°,  Translation: 0.5m
```
Errors grow gradually, model handles scale changes well.

### Poor Performance
```
Distance: 10m  → Rotation: 5°,  Translation: 0.3m
Distance: 50m  → Rotation: 25°, Translation: 2.1m
Distance: 100m → Rotation: 45°, Translation: 5.8m
```
Errors explode rapidly, model struggles with scale.

## 🚦 Next Steps

### Immediate Actions
1. ✅ Run `python list_scenes.py` to see available scenes
2. ✅ Run `python multi_scale_eval.py --scene 0000` for first evaluation
3. ✅ Check `multi_scale_results/` for output files

### Short Term (Next Hour)
4. Try different scenes to see variation
5. Experiment with `--num_pairs` parameter
6. Test `--base_image` to use specific images

### Medium Term (This Week)
7. Evaluate multiple scenes systematically
8. Analyze CSV files for trends
9. Compare results across different scene types
10. Generate summary statistics

### Long Term (Research)
11. Correlate with scene characteristics
12. Compare with other methods
13. Write up findings
14. Create visualizations for paper

## 🔧 Troubleshooting

### "Scene not found"
```bash
# Check valid scenes
python list_scenes.py
```

### "CUDA out of memory"
```bash
# Use memory-efficient mode
python multi_scale_eval.py --scene 0000 --memory_efficient
```

### "No images found"
```bash
# Verify scene structure
ls multi-scale-dataset/aerial-megadepth/0000/sfm_output_localization/sfm_superpoint+superglue/localized_dense_metric/images/
```

### Need help?
- Quick answers: `QUICK_REFERENCE.md`
- Detailed help: `README_MULTI_SCALE.md`
- Technical issues: `MULTI_SCALE_EVALUATION.md`

## 🎯 Your Question Answered

**Your requirement**: 
> "Evaluate how feedforward models like mapanything perform when distance between images of the same scene increases"

**What this system does**:
1. ✅ Takes aerial-megadepth dataset with COLMAP poses
2. ✅ Selects image pairs at varying distances
3. ✅ Runs MapAnything to predict poses
4. ✅ Compares with ground truth
5. ✅ Outputs CSV showing: distance vs rotation error vs translation error
6. ✅ Generates plots: distance vs pose accuracy
7. ✅ Allows user to choose scene via `--scene` flag

**All requirements met!** ✓

## 🎉 You're All Set!

Everything is ready. Just run:

```bash
python multi_scale_eval.py --scene 0000
```

Then check the results in `multi_scale_results/`.

**Questions?** Check the documentation files listed above.

**Good luck with your evaluation!** 🚀

---

**Pro tip**: Start with a small scene and `--num_pairs 10` to verify everything works, then scale up to your full evaluation.

