#!/bin/bash
#SBATCH --job-name=p2v-cls-sc
#SBATCH --no-requeue
#SBATCH --time=2-00:00
#SBATCH --begin=now
#SBATCH --signal=TERM@120
#SBATCH --output=slurm_logs/%j_%n_%x.txt

set -e

python -m point2vec.tasks.classification fit --config "configs/classification/scanobjectnn.yaml" --config "configs/wandb/classification_scanobjectnn.yaml" "$@"