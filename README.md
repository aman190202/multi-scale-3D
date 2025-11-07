Interact session
```bash
interact -n 4 -t 12:00:00 -m 24g -q gpu -g 1
export SCRATCH=/users/aagar133/scratch/multi-scale-3D
export DATASET=/users/aagar133/scratch/multi-scale-3D/satellite_to_street_dataset
export SCENE=CIT-Brown-University

module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5
module load miniconda3/23.11.0s
module load ffmpeg

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate $SCRATCH/mscale_env
```

