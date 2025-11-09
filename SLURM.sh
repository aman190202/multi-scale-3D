#!/bin/bash
#SBATCH --job-name=mscale
#SBATCH --output=logs/mscale_%j.out
#SBATCH --error=logs/mscale_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8

# -----------------------------
# Environment setup
# -----------------------------

export SCRATCH=/users/aagar133/scratch/multi-scale-3D
module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5
module load miniconda3/23.11.0s
module load ffmpeg

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# -----------------------------
# Installation 
# -----------------------------
# Undo all the commented lines to create a first time setup

# conda create --prefix $SCRATCH/mscale_env python=3.11 --solver classic -y
conda activate $SCRATCH/mscale_env
# conda install -c conda-forge colmap=3.11.0

# pip install pandas protobuf packaging pillow
# pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
# conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit --solver classic -y
# pip install pillow
# pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation
# pip install --user huggingface_hub
# pip install nerfstudio



# -----------------------------
# DATASET INSTALLATION
# -----------------------------
# export HF_TOKEN=hf_yourtokenhere
# python download.py

export SCRATCH=/users/aagar133/scratch/multi-scale-3D
export DATASET=/users/aagar133/scratch/multi-scale-3D/satellite_to_street_dataset
export SCENE=CIT-Brown-University


# CUDA_VISIBLE_DEVICES=0 ns-process-data images --data $DATASET/$SCENE/footage --output-dir $DATASET/$SCENE/


# -----------------------------
# MAP ANYTHING
# -----------------------------
# pip install "git+https://github.com/facebookresearch/map-anything.git@fde8425513178bb4f89fba9828193e6be3ece248#egg=map-anything[all]"

# Using a stride 4 will only select every 4th image for map-anything inference ; we are doing this to not overload the memory 
python map-anything-inference.py --scene_dir $DATASET/$SCENE-MapAnything --memory_efficient_inference --stride 1


python /users/aagar133/scratch/multi-scale-3D/map-anything/scripts/make_pointcloud_pairs_gif.py  \
  --images_dir $DATASET/$SCENE/images \
  --stride 1 \
  --fps 5 \
  --conf_percentile 0 \
  --filter_black_bg \
  --filter_white_bg \
  --end_index 450 \
  --points_per_frame 40000 \
  --output pointcloud_pairs.gif