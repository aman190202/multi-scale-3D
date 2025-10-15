# Multi-Scale 3D - Pose Prediction Evaluation

## 🆕 Multi-Scale Evaluation Tool

This repository includes a comprehensive evaluation tool for assessing how feedforward pose prediction models (like MapAnything) perform as the distance between images increases.

### Quick Start

```bash
# List available scenes
python list_scenes.py

# Run evaluation on a scene
python multi_scale_eval.py --scene 0000

# View detailed documentation
cat README_MULTI_SCALE.md
```

### Key Features
- ✅ Automatic COLMAP data parsing from aerial-megadepth dataset
- ✅ Distance-based image pair selection
- ✅ Pose prediction vs ground truth comparison
- ✅ CSV export and visualization plots
- ✅ 137 scenes with 132,000+ images available

**Complete documentation**: See [README_MULTI_SCALE.md](README_MULTI_SCALE.md)

---

## SLURM guide

```bash
interact -q gpu -n 8 -m 16g -t 01:00:00 -g 1 -f a5000
```
a5000 has sm_86 architecture

```bash
module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
module load gcc/10
module load python/3.11
export TORCH_CUDA_ARCH_LIST="8.6"

```

## Map Anything
```bash
python -m venv env
source env/bin/activate
pip install  "git+https://github.com/facebookresearch/map-anything.git@44cbc5a4a3960cab50c449797afef200b88b94b7#egg=mapanything"
```

## Datasets
1. University 1652
2. Aerial Mega-depth