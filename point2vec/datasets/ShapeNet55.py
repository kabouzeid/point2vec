import json
import os
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
from pytorch3d.datasets.shapenet.shapenet_core import SYNSET_DICT_DIR
from torch.utils.data import DataLoader, Dataset


class ShapeNet55(Dataset):
    """
    Version of ShapeNet used in Point-BERT (https://github.com/lulutang0608/Point-BERT/tree/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52)
    for pretraining the model.
    There are some minor modifications to the original code.
    """

    def __init__(self, root, split="train"):
        assert split in ["train", "train_75", "train_50", "train_25", "test"]

        # Synset dictionary mapping synset offsets to corresponding labels.
        with open(
            os.path.join(SYNSET_DICT_DIR, "shapenet_synset_dict_v2.json"), "r"
        ) as read_dict:
            self.synset_dict: Dict = json.load(read_dict)
        assert len(self.synset_dict) == 55

        self._label_names = sorted(self.synset_dict.keys())
        self.label2synset_id = dict(enumerate(self._label_names))
        self.synset_id2label = {v: k for k, v in self.label2synset_id.items()}

        pc_folder_name = "shapenet_pc"
        split_file_name = f"{split}.txt"

        self.pc_path = os.path.join(root, pc_folder_name)

        # reading split file names
        with open(os.path.join(root, split_file_name), "r") as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            synset_id = line.split("-")[0]
            # model_id = line.split("-")[1].split(".")[0]

            self.file_list.append(
                {
                    "label": self.synset_id2label[synset_id],
                    # "synset_id": synset_id,
                    # "model_id": model_id,
                    "file_path": line,
                }
            )

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        label = sample["label"]
        pc_path = os.path.join(self.pc_path, sample["file_path"])
        data = np.load(pc_path).astype(np.float32)
        return data, label

    def __len__(self):
        return len(self.file_list)

    @property
    def label_names(self) -> List[str]:
        return self._label_names


# uses the npz files created by `./process/shapenet_in_memory.py`
class ShapeNet55Npz(Dataset):
    def __init__(self, root, split="train"):
        assert split in ["train", "test"]

        with open(
            os.path.join(SYNSET_DICT_DIR, "shapenet_synset_dict_v2.json"), "r"
        ) as read_dict:
            self.synset_dict: Dict = json.load(read_dict)
        assert len(self.synset_dict) == 55
        self._label_names = sorted(self.synset_dict.keys())

        with np.load(os.path.join(root, f"shapenet_{split}.npz")) as f:
            self.data = f["data"]
            self.labels = f["labels"]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

    @property
    def label_names(self) -> List[str]:
        return self._label_names


class ShapeNet55DataModule(pl.LightningDataModule):
    """
    size: 52470
    train: 41952
    test: 10518
    """

    def __init__(
        self,
        data_dir: str = "./data/ShapeNet55",
        split: Optional[str | int] = None,
        batch_size: int = 512,
        num_workers: int = 8,
        in_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_workers = 0 if in_memory else num_workers
        self.persistent_workers = True if self.num_workers > 0 else False

    def setup(self, stage: Optional[str] = None):
        split: Optional[str] = self.hparams.split  # type: ignore
        if self.hparams.in_memory:  # type: ignore
            assert split is None
            self.train_dataset = ShapeNet55Npz(self.hparams.data_dir, split="train")  # type: ignore
            self.test_dataset = ShapeNet55Npz(self.hparams.data_dir, split="test")  # type: ignore
        else:
            self.train_dataset = ShapeNet55(self.hparams.data_dir, split="train" if split is None else "train_{}".format(split))  # type: ignore
            self.test_dataset = ShapeNet55(self.hparams.data_dir, split="test")  # type: ignore

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,  # type: ignore
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    @property
    def num_classes(self):
        return 55

    @property
    def label_names(self) -> List[str]:
        return self.train_dataset.label_names
