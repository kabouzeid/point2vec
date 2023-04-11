#!/bin/bash
#SBATCH --job-name=p2v-pse-snp
#SBATCH --no-requeue
#SBATCH --time=2-00:00
#SBATCH --begin=now
#SBATCH --signal=TERM@120
#SBATCH --output=slurm_logs/%j_%n_%x.txt

set -e

python -m point2vec.tasks.part_segmentation fit --config "configs/part_segmentation/shapenetpart.yaml" --config "configs/wandb/part_segmentation_shapenet_part.yaml" "$@"
