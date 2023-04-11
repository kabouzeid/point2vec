from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from point2vec.datasets import (  # allow shorthand notation
    ModelNet40FewShotDataModule,
    ModelNet40Ply2048DataModule,
    ScanObjectNNDataModule,
    ShapeNet55DataModule,
)
from point2vec.models import Point2VecClassification

if __name__ == "__main__":
    cli = LightningCLI(
        Point2VecClassification,
        trainer_defaults={
            "default_root_dir": "artifacts",
            "accelerator": "gpu",
            "devices": 1,
            "precision": 16,
            "max_epochs": 300,
            "track_grad_norm": 2,
            "callbacks": [
                LearningRateMonitor(),
                ModelCheckpoint(save_on_train_epoch_end=True),
                ModelCheckpoint(
                    filename="{epoch}-{step}-{val_acc:.4f}",
                    monitor="val_acc",
                    mode="max",
                ),
            ],
        },
        seed_everything_default=0,
        save_config_callback=None,  # https://github.com/Lightning-AI/lightning/issues/12028#issuecomment-1088325894
    )
