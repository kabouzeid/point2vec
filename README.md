# point2vec

Self-Supervised Representation Learning on Point Clouds

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point2vec-for-self-supervised-representation/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=point2vec-for-self-supervised-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point2vec-for-self-supervised-representation/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=point2vec-for-self-supervised-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point2vec-for-self-supervised-representation/3d-part-segmentation-on-shapenet-part)](https://paperswithcode.com/sota/3d-part-segmentation-on-shapenet-part?p=point2vec-for-self-supervised-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point2vec-for-self-supervised-representation/few-shot-3d-point-cloud-classification-on-3)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-3?p=point2vec-for-self-supervised-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point2vec-for-self-supervised-representation/few-shot-3d-point-cloud-classification-on-4)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-4?p=point2vec-for-self-supervised-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point2vec-for-self-supervised-representation/few-shot-3d-point-cloud-classification-on-1)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-1?p=point2vec-for-self-supervised-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point2vec-for-self-supervised-representation/few-shot-3d-point-cloud-classification-on-2)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-2?p=point2vec-for-self-supervised-representation)

[[`Paper`](https://arxiv.org/abs/2303.16570)] [[`Project`](https://point2vec.ka.codes)] [[`BibTeX`](#citing-point2vec)]

![architecture](https://user-images.githubusercontent.com/7303830/229210206-3df19b2c-f0bf-46ee-b0c2-d1ec1a3123d4.png)

## Installation

### 1. Dependencies

- Python 3.10.4
- CUDA 11.6
- cuDNN 8.4.0
- GCC >= 6 and <= 11.2.1

```bash
pip install -U pip wheel
pip install torch torchvision -c requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

### 2. Datasets

See [DATASETS.md](DATASETS.md) for download instructions.

### 3. Check (optional)

```bash
python -m point2vec.datasets.process.check # check if datasets are complete
./scripts/test.sh # check if training works
```

## Model Zoo

| Type                         | Dataset      | Evaluation                          | Config                                                                                       | Checkpoint                                                                                                                                    |
| ---------------------------- | ------------ | ----------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Point2vec pre-trained        | ShapeNet     | -                                   | [config](configs/pretraining/shapenet.yaml)                                                  | [checkpoint](https://github.com/kabouzeid/point2vec/releases/download/paper/pre_point2vec-epoch.799-step.64800.ckpt)                          |
| Classification fine-tuned    | ModelNet40   | **94.65** / **94.77** (OA / Voting) | [A](configs/classification/modelnet40.yaml) & [B](configs/classification/_pretrained.yaml)   | [checkpoint](https://github.com/kabouzeid/point2vec/releases/download/paper/fine_modelnet40-epoch.125-step.38682-val_acc.0.9465.ckpt)         |
| Classification fine-tuned    | ScanObjectNN | **87.47** (OA)                      | [A](configs/classification/scanobjectnn.yaml) & [B](configs/classification/_pretrained.yaml) | [checkpoint](https://github.com/kabouzeid/point2vec/releases/download/paper/fine_scanobjectnn-epoch.146-step.52332-val_acc.0.8747.ckpt)       |
| Part segmentation fine-tuned | ShapeNetPart | **84.59** (Cat. mIoU)               | [config](configs/part_segmentation/shapenetpart.yaml)                                        | [checkpoint](https://github.com/kabouzeid/point2vec/releases/download/paper/fine_shapenetpart-epoch.288-step.252586-val_cat_miou.0.8459.ckpt) |

## Reproducing the results from the paper

The scripts in this section use Weights & Biases for logging, so it's important to log in once with `wandb login` before running them.
Checkpoints will be saved to the `artifacts` directory.

**A note on reproducibility:**
While reproducing our results on most datasets is straightforward, achieving the same test accuracy on ModelNet40 is more complicated due to the high variance between runs (see also https://github.com/Pang-Yatian/Point-MAE/issues/5#issuecomment-1074886349, https://github.com/ma-xu/pointMLP-pytorch/issues/1#issuecomment-1062563404, https://github.com/CVMI-Lab/PAConv/issues/9#issuecomment-886612074).
To obtain comparable results on ModelNet40, you will likely need to experiment with a few different seeds.
However, if you can precisely replicate our test environment, including installing CUDA 11.6, cuDNN 8.4.0, Python 3.10.4, and the dependencies listed in the `requirements.txt` file, as well as using a Volta GPU (e.g. Nvidia V100), you should be able to replicate our experiments exactly.
Using our _exact_ environment is necessary to ensure that you obtain the same random state during training, as a seed alone does _not_ guarantee reproducibility across different environments.

### Point2vec pre-training on ShapeNet

```bash
./scripts/pretraining_shapenet.bash --data.in_memory true
```

<details>
<summary>Training curve</summary>

![pre-training](https://user-images.githubusercontent.com/7303830/230649527-8ccd2af9-6c3f-4495-8c58-5854aa2a3304.png)

</details>

[[`Checkpoint`](https://github.com/kabouzeid/point2vec/releases/download/paper/pre_point2vec-epoch.799-step.64800.ckpt)]

### Classification fine-tuning on ScanObjectNN

Replace `XXXXXXXX` with the `WANDB_RUN_ID` from the pre-training run, or use the checkpoint from the model zoo.

```bash
./scripts/classification_scanobjectnn.bash --config configs/classification/_pretrained.yaml --model.pretrained_ckpt_path artifacts/point2vec-Pretraining-ShapeNet/XXXXXXXX/checkpoints/epoch=799-step=64800.ckpt
```

<details>
<summary>Training curve</summary>

![scanobjectnn](https://user-images.githubusercontent.com/7303830/230649569-b68d5389-3164-4b51-a31c-e512c3d9234b.png)

</details>

[[`Checkpoint`](https://github.com/kabouzeid/point2vec/releases/download/paper/fine_scanobjectnn-epoch.146-step.52332-val_acc.0.8747.ckpt)]

### Classification fine-tuning on ModelNet40

Replace `XXXXXXXX` with the `WANDB_RUN_ID` from the pre-training run, or use the checkpoint from the model zoo.

```bash
./scripts/classification_modelnet40.bash --config configs/classification/_pretrained.yaml --model.pretrained_ckpt_path artifacts/point2vec-Pretraining-ShapeNet/XXXXXXXX/checkpoints/epoch=799-step=64800.ckpt --seed_everything 1
```

<details>
<summary>Training curve</summary>

![modelnet40](https://user-images.githubusercontent.com/7303830/230651790-3fa5c959-5a7f-4e35-b219-1424a21b9c2d.png)

</details>

[[`Checkpoint`](https://github.com/kabouzeid/point2vec/releases/download/paper/fine_modelnet40-epoch.125-step.38682-val_acc.0.9465.ckpt)]

### Voting on ModelNet40

Replace `XXXXXXXX` with the `WANDB_RUN_ID` from the fine-tuning run, and `epoch=XXX-step=XXXXX-val_acc=0.XXXX.ckpt` with the best checkpoint from that run, or use the checkpoint from the model zoo.

```bash
./scripts/voting_modelnet40.bash --finetuned_ckpt_path artifacts/point2vec-Pretraining-ShapeNet/XXXXXXXX/checkpoints/epoch=XXX-step=XXXXX-val_acc=0.XXXX.ckpt
```

<details>
<summary>Voting Process</summary>

![voting](https://user-images.githubusercontent.com/7303830/230651975-d356919d-b1ba-40a8-9d92-60b107f10e8d.png)

</details>

### Classification fine-tuning on ModelNet Few-Shot

Replace `XXXXXXXX` with the `WANDB_RUN_ID` from the pre-training run, or use the checkpoint from the model zoo.
You may also pass e.g. `--data.way 5` or `--data.shot 20` to select the desired m-way&ndash;n-shot setting.

```bash
for i in $(seq 0 9);
do
    SLURM_ARRAY_TASK_ID=$i ./scripts/classification_modelnet_fewshot.bash --model.pretrained_ckpt_path artifacts/point2vec-Pretraining-ShapeNet/XXXXXXXX/checkpoints/epoch=799-step=64800.ckpt
done
```

### Part segmentation fine-tuning on ShapeNetPart

Replace `XXXXXXXX` with the `WANDB_RUN_ID` from the pre-training run, or use the checkpoint from the model zoo.

```bash
./scripts/part_segmentation_shapenetpart.bash --model.pretrained_ckpt_path artifacts/point2vec-Pretraining-ShapeNet/XXXXXXXX/checkpoints/epoch=799-step=64800.ckpt
```

<details>
<summary>Training curve</summary>

![shapenetpart](https://user-images.githubusercontent.com/7303830/230651811-bd029146-aa30-4903-b011-5a0a8475cdda.png)

</details>

[[`Checkpoint`](https://github.com/kabouzeid/point2vec/releases/download/paper/fine_shapenetpart-epoch.288-step.252586-val_cat_miou.0.8459.ckpt)]

### Baselines

<details>

<summary>Expand</summary>

#### Data2vec&ndash;pc

Replace the pre-training step with:

```bash
./scripts/pretraining_shapenet.bash --data.in_memory true --model.learning_rate 2e-3 --model.decoder false --trainer.devices 2 --data.batch_size 1024 --model.fix_estimated_stepping_batches 16000
```

If you only have a single GPU (and enough VRAM), you may replace `--trainer.devices 2 --data.batch_size 1024 --model.fix_estimated_stepping_batches 16000` with `--data.batch_size 2048`.

[[`Checkpoint`](https://github.com/kabouzeid/point2vec/releases/download/paper/pre_data2vec-epoch.799-step.16000.ckpt)]

#### From scratch

Skip the pre-training step, and omit all occurences of `--config configs/classification/_pretrained.yaml` and `--model.pretrained_ckpt_path ...`.

</details>

## Visualization

![representations](https://user-images.githubusercontent.com/7303830/230671271-b8bb7dbb-1d21-4391-93f9-631d645587c0.png)

We use PCA to project the learned representations into RGB space.
Both a random initialization and data2vec&ndash;pc pre-training show a fairly strong positional bias, whereas point2vec exhibits a stronger semantic grouping without being trained on downstream dense prediction tasks.

## Citing point2vec

If you use point2vec in your research, please use the following BibTeX entry.

```
@article{abouzeid2023point2vec,
  title={Point2Vec for Self-Supervised Representation Learning on Point Clouds},
  author={Abou Zeid, Karim and Schult, Jonas and Hermans, Alexander and Leibe, Bastian},
  journal={arXiv preprint arXiv:2303.16570},
  year={2023},
}
```
