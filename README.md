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

## Observations : 

	1.	The COLMAP reconstruction reports the most accurate angular separation between the camera poses of `frame_00001.jpeg` and `frame_000449.jpeg`, measuring 89.8 degrees.
	2.	When the same dataset is re-inferred using every third image (to reduce memory usage), the estimated angular difference between `frame_00001.jpeg` and `frame_000449.jpeg` decreases to 83.6529 degrees.
	3.	A subsequent inference under the same every-third-image sampling yields an angular difference of 82.604 degrees between `frame_00001.jpeg` and `frame_000448.jpeg`.