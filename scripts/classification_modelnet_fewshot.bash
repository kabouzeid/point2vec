#!/bin/bash
#SBATCH --job-name=p2v-cls-few
#SBATCH --no-requeue
#SBATCH --time=2-00:00
#SBATCH --begin=now
#SBATCH --signal=TERM@120
#SBATCH --output=slurm_logs/%A_%a_%n_%x.txt
#SBATCH --array=0-9

set -e

python -m point2vec.tasks.classification fit --config "configs/classification/modelnet_fewshot.yaml" --config "configs/wandb/classification_modelnet_fewshot.yaml" --data.fold "${SLURM_ARRAY_TASK_ID}" "$@"