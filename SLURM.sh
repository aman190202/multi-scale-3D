#!/bin/bash
#SBATCH --job-name=mscale
#SBATCH --output=logs/mscale_%j.out
#SBATCH --error=logs/mscale_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4

# -----------------------------
# Environment setup
# -----------------------------

SCRATCH=/users/aagar133/scratch/multi-scale-3D

module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5
module load miniconda3/23.11.0s
module load ffmpeg

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# -----------------------------
# Installation
# -----------------------------

# conda create --prefix $SCRATCH/mscale_env python=3.11 --solver classic -y
conda activate $SCRATCH/mscale_env

# pip install pandas protobuf packaging
# pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
# conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
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

# -----------------------------
# COLMAP SETUP 
# -----------------------------


# singularity pull docker://colmap/colmap:latest
# mkdir -p ~/bin
# cat <<'EOF' > ~/bin/colmap
# #!/bin/bash
# singularity exec --nv $SCRATCH/multi-scale-3D/colmap_latest.sif colmap "$@"
# EOF
# chmod +x ~/bin/colmap
# export PATH=$HOME/bin:$PATH

ns-process-data images --data $DATASET/$SCENE/footage --output_dir $DATASET/$SCENE/nerfstudio --verbose