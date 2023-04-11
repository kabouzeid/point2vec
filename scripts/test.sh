#!/bin/bash

set -e

echo && echo "Testing pre-training ShapeNet... (validation step might take a few minutes)" && echo
python -m point2vec                             fit --config configs/pretraining/shapenet.yaml            --trainer.fast_dev_run true --data.batch_size 128 "$@"

echo && echo "Testing finetuning ModelNet40..." && echo
python -m point2vec.tasks.classification        fit --config configs/classification/modelnet40.yaml       --trainer.fast_dev_run true "$@"
echo && echo "Testing finetuning ScanObjectNN..." && echo
python -m point2vec.tasks.classification        fit --config configs/classification/scanobjectnn.yaml     --trainer.fast_dev_run true "$@"
echo && echo "Testing finetuning ModelNet Few-Shot..." && echo
python -m point2vec.tasks.classification        fit --config configs/classification/modelnet_fewshot.yaml --trainer.fast_dev_run true "$@"

echo && echo "Testing finetuning ShapeNetPart..." && echo
python -m point2vec.tasks.part_segmentation     fit --config configs/part_segmentation/shapenetpart.yaml  --trainer.fast_dev_run true "$@"

echo && echo "All tests passed." && echo