import torch
import torch.nn.functional as F
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from pytorch_lightning.loggers import WandbLogger

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
        parser.add_argument("--vote_num", type=int, default=10)
        parser.add_argument("--vote_iterations", type=int, default=300)


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

    for logger in cli.trainer.loggers:
        if isinstance(logger, WandbLogger):
            logger.experiment.define_metric("val_vote_acc", summary="max")

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

    best_acc = 0
    for i in range(cli.config.vote_iterations):
        acc = vote(cli)
        if acc > best_acc:
            best_acc = acc
        print(
            f"{i}/{cli.config.vote_iterations} Vote Acc: {acc * 100:.2f}%, Best Vote Acc: {best_acc * 100:.2f}%"
        )
        for logger in cli.trainer.loggers:
            logger.log_metrics({"val_vote_acc": acc}, step=i)


def vote(cli: CLI) -> float:
    cli.model.eval()
    vote_pred_list = []
    label_list = []
    for (points, label) in iter(cli.datamodule.val_dataloader()):  # type: ignore
        points = points.cuda()
        label = label.cuda()

        with torch.no_grad():
            logits_list = []
            points = cli.model.val_transformations(points)
            for _ in range(cli.config.vote_num):
                logits = cli.model(points)
                logits_list.append(F.softmax(logits, dim=-1))
            mean_logits = torch.mean(torch.stack(logits_list), dim=0)
            vote_pred = torch.max(mean_logits, dim=-1).indices

        vote_pred_list.append(vote_pred.cpu())
        label_list.append(label.cpu())

    vote_pred = torch.cat(vote_pred_list, dim=0)  # (N, 768)
    label = torch.cat(label_list, dim=0)  # (N,)

    return (torch.sum(vote_pred == label) / label.shape[0]).item()


if __name__ == "__main__":
    main()
