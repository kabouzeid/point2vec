# copied and modified from https://github.dev/Pang-Yatian/Point-MAE/blob/fd5a476408220b4b31032eee62b5e89d051481a2/segmentation/dataset.py

import json
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class ShapeNetPart(Dataset):
    def __init__(
        self,
        root="./data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
        num_points=2048,
        split="train",
        normals=False,
    ):
        self.num_points = num_points
        self.root = root
        self.catfile = os.path.join(self.root, "synsetoffset2category.txt")
        self.cat = {}
        self.normals = normals

        with open(self.catfile, "r") as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        self.meta = {}
        with open(
            os.path.join(
                self.root, "train_test_split", "shuffled_train_file_list.json"
            ),
            "r",
        ) as f:
            train_ids = set([str(d.split("/")[2]) for d in json.load(f)])
        with open(
            os.path.join(self.root, "train_test_split", "shuffled_val_file_list.json"),
            "r",
        ) as f:
            val_ids = set([str(d.split("/")[2]) for d in json.load(f)])
        with open(
            os.path.join(self.root, "train_test_split", "shuffled_test_file_list.json"),
            "r",
        ) as f:
            test_ids = set([str(d.split("/")[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == "trainval":
                fns = [
                    fn
                    for fn in fns
                    if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))
                ]
            elif split == "train":
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == "val":
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == "test":
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print("Unknown split: %s. Exiting.." % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = os.path.splitext(os.path.basename(fn))[0]
                self.meta[item].append(os.path.join(dir_point, token + ".txt"))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

    def __getitem__(self, index):
        fn = self.datapath[index]
        cat = self.datapath[index][0]
        label = self.classes[cat]
        label = label
        data = np.loadtxt(fn[1]).astype(np.float32)
        if not self.normals:
            points = data[:, :3]
        else:
            points = data[:, :6]
        seg_labels = data[:, -1].astype(np.int64)

        # TODO: find a better way to do this
        choice = np.random.choice(len(seg_labels), self.num_points, replace=True)
        # resample
        points = points[choice]
        seg_labels = seg_labels[choice]

        return points, seg_labels, label

    def __len__(self):
        return len(self.datapath)


class ShapeNetPartDataModule(pl.LightningDataModule):
    """ """

    def __init__(
        self,
        data_dir: str = "./data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
        batch_size: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self._category_to_seg_classes = {
            "Earphone": [16, 17, 18],
            "Motorbike": [30, 31, 32, 33, 34, 35],
            "Rocket": [41, 42, 43],
            "Car": [8, 9, 10, 11],
            "Laptop": [28, 29],
            "Cap": [6, 7],
            "Skateboard": [44, 45, 46],
            "Mug": [36, 37],
            "Guitar": [19, 20, 21],
            "Bag": [4, 5],
            "Lamp": [24, 25, 26, 27],
            "Table": [47, 48, 49],
            "Airplane": [0, 1, 2, 3],
            "Pistol": [38, 39, 40],
            "Chair": [12, 13, 14, 15],
            "Knife": [22, 23],
        }

        # inverse mapping
        self._seg_class_to_category = {}
        for cat in self._category_to_seg_classes.keys():
            for cls in self._category_to_seg_classes[cat]:
                self._seg_class_to_category[cls] = cat

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ShapeNetPart(self.hparams.data_dir, split="trainval")  # type: ignore
        self.test_dataset = ShapeNetPart(self.hparams.data_dir, split="test")  # type: ignore

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,  # type: ignore
            shuffle=True,
            drop_last=True,  # type: ignore
            num_workers=8,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=8,
            persistent_workers=True,
        )

    @property
    def num_classes(self):
        return 16

    @property
    def category_to_seg_classes(self):
        return self._category_to_seg_classes

    @property
    def seg_class_to_category(self):
        return self._seg_class_to_category

    @property
    def num_seg_classes(self):
        return 50
