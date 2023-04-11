#!/bin/bash
#SBATCH --job-name=p2v-vot-mo
#SBATCH --no-requeue
#SBATCH --time=2-00:00
#SBATCH --begin=now
#SBATCH --signal=TERM@120
#SBATCH --output=slurm_logs/%j_%n_%x.txt

set -e

# -u for unbuffered output
python -u -m point2vec.eval.voting --config "configs/classification/modelnet40.yaml" --config "configs/wandb/voting_modelnet40.yaml" --data.batch_size 256 "$@"