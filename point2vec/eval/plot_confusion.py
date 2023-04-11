from pathlib import Path

import torch
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from point2vec.datasets import (  # allow shorthand notation
    ModelNet40FewShotDataModule,
    ModelNet40Ply2048DataModule,
    ScanObjectNNDataModule,
    ShapeNet55DataModule,
)
from point2vec.models import Point2VecClassification
from point2vec.utils.checkpoint import extract_model_checkpoint


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--finetuned_ckpt_path", type=str, required=True)


def plot_confusion_matrix(label, pred, label_names):
    from matplotlib import pyplot as plt

    # import seaborn as sns
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    # import pandas as pd
    # data = pd.DataFrame(m)
    # data = data.set_axis(label_names, axis=0)
    # data = data.set_axis(label_names, axis=1)
    # data = data.rename_axis("Ground truth", axis=0)
    # data = data.rename_axis("Prediction", axis=1)
    # plt.figure(figsize=(12, 12))
    # ax = sns.heatmap(data, cmap="viridis", linewidth=1)
    # ax.set_xticks([i + 0.5 for i in range(len(label_names))])
    # ax.set_xticklabels(label_names, rotation=45, ha="right")
    # ax.set_yticks([i + 0.5 for i in range(len(label_names))])
    # ax.set_yticklabels(label_names)
    # ax.set_xlabel("Prediction")
    # ax.set_ylabel("Ground truth")
    # ax.set_aspect("equal", adjustable="box")
    # plt.savefig("figures/confusion_matrix.pdf")
    # plt.clf()

    def plot(m):
        plt.figure(figsize=(9, 9))
        disp = ConfusionMatrixDisplay(m, display_labels=label_names)
        disp.plot(
            xticks_rotation="vertical",
            include_values=False,
            ax=plt.gca(),
            cmap="Reds",
            colorbar=False,
        )
        disp.figure_.colorbar(disp.im_, fraction=0.046, pad=0.04)

    dir = Path("figures/confusion_matrix")
    dir.mkdir(exist_ok=True)

    plot(confusion_matrix(label, pred))
    plt.savefig(dir / "confusion_matrix.pdf")
    plt.clf()

    plot(confusion_matrix(label, pred, normalize="true"))
    plt.savefig(dir / "confusion_matrix_norm_true.pdf")
    plt.clf()

    plot(confusion_matrix(label, pred, normalize="pred"))
    plt.savefig(dir / "confusion_matrix_norm_pred.pdf")
    plt.clf()


def main():
    cli = CLI(
        Point2VecClassification,
        trainer_defaults={
            "default_root_dir": "artifacts",
            # defaults below don't matter here, but will shut up the warnings
            "accelerator": "gpu",
            "devices": 1,
        },
        seed_everything_default=0,
        save_config_callback=None,  # https://github.com/Lightning-AI/lightning/issues/12028#issuecomment-1088325894
        run=False,
    )

    assert isinstance(cli.model, Point2VecClassification)

    # our model needs the trainer to lookup the datamodule
    cli.model.trainer = cli.trainer
    cli.model.trainer.datamodule = cli.datamodule  # type: ignore

    cli.datamodule.setup()
    cli.model.setup()
    cli.model = cli.model.cuda()

    checkpoint = extract_model_checkpoint(cli.config.finetuned_ckpt_path)
    missing_keys, unexpected_keys = cli.model.load_state_dict(checkpoint, strict=False)  # type: ignore
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    cli.model.eval()
    pred_list = []
    label_list = []
    points_list = []
    for (points, label) in iter(cli.datamodule.val_dataloader()):  # type: ignore
        points = points.cuda()
        label = label.cuda()

        with torch.no_grad():
            points = cli.model.val_transformations(points)
            logits, _ = cli.model(points)
            pred = torch.max(logits, dim=-1).indices

        pred_list.append(pred.cpu())
        label_list.append(label.cpu())
        points_list.append(points.cpu())

    pred = torch.cat(pred_list, dim=0)  # (N,)
    label = torch.cat(label_list, dim=0)  # (N,)
    points = torch.cat(points_list, dim=0)  # (N, 1024, 3)

    plot_confusion_matrix(label.numpy(), pred.numpy(), cli.datamodule.label_names)
    # torch.save(pred, "figures/confusion_matrix/confusion_pred.pt")
    # torch.save(label, "figures/confusion_matrix/confusion_label.pt")
    # torch.save(points, "figures/confusion_matrix/confusion_points.pt")


if __name__ == "__main__":
    main()
