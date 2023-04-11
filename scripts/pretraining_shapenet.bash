#!/bin/bash
#SBATCH --job-name=p2v-pre-sh
#SBATCH --no-requeue
#SBATCH --time=2-00:00
#SBATCH --begin=now
#SBATCH --signal=TERM@120
#SBATCH --output=slurm_logs/%j_%n_%x.txt

set -e

python -m point2vec fit --config "configs/pretraining/shapenet.yaml" --config "configs/wandb/pretraining_shapenet.yaml" "$@"