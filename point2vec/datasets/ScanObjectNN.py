from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class ScanObjectNN(Dataset):
    def __init__(self, root, split="training", perturbed=True):
        assert split == "training" or split == "test"
        file_name = (
            "_objectdataset_augmentedrot_scale75.h5"
            if perturbed
            else "_objectdataset.h5"
        )
        h5_name = Path(root) / (split + file_name)
        with h5py.File(h5_name, mode="r") as f:
            self.data = f["data"][:].astype(np.float32)  # type: ignore
            self.label = f["label"][:].astype(np.int64)  # type: ignore

    def __len__(self):
        return self.data.shape[0]  # type: ignore

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class ScanObjectNNDataModule(pl.LightningDataModule):
    """
    size: 14298
    train: 11416
    test: 2882
    """

    def __init__(
        self,
        data_dir: str = "./data/ScanObjectNN",
        split: str = "main_split",
        perturbed: bool = True,
        batch_size: int = 32,
        drop_last: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ScanObjectNN(Path(self.hparams.data_dir) / self.hparams.split, split="training", perturbed=self.hparams.perturbed)  # type: ignore
        self.test_dataset = ScanObjectNN(Path(self.hparams.data_dir) / self.hparams.split, split="test", perturbed=self.hparams.perturbed)  # type: ignore

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,  # type: ignore
            shuffle=True,
            drop_last=self.hparams.drop_last,  # type: ignore
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,  # type: ignore
        )

    @property
    def num_classes(self):
        return 15

    @property
    def label_names(self) -> List[str]:
        return [
            "bag",
            "bin",
            "box",
            "cabinet",
            "chair",
            "desk",
            "display",
            "door",
            "shelf",
            "table",
            "bed",
            "pillow",
            "sink",
            "sofa",
            "toilet",
        ]
