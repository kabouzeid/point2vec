#!/bin/bash
#SBATCH --job-name=p2v-pre-mo
#SBATCH --no-requeue
#SBATCH --time=2-00:00
#SBATCH --begin=now
#SBATCH --signal=TERM@120
#SBATCH --output=slurm_logs/%j_%n_%x.txt

set -e

python -m point2vec fit --config "configs/pretraining/modelnet40.yaml" --config "configs/wandb/pretraining_modelnet40.yaml" "$@"