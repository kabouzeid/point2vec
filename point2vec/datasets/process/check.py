"""
Check that the data dir contains the correct number of files for each dataset.
"""
if __name__ == "__main__":
    from typing import Tuple, Type

    import pytorch_lightning as pl

    from point2vec.datasets import (
        ModelNet40FewShotDataModule,
        ModelNet40Ply2048DataModule,
        ScanObjectNNDataModule,
        ShapeNet55DataModule,
        ShapeNetPartDataModule,
    )

    def dataset_lens(
        data_module_type: Type[pl.LightningDataModule], **kwargs
    ) -> Tuple[int, int]:
        print(f"Checking {data_module_type.__name__} {kwargs}...")
        data_module = data_module_type(**kwargs)
        data_module.setup("fit")
        return len(data_module.train_dataloader().dataset), len(data_module.val_dataloader().dataset)  # type: ignore

    assert dataset_lens(ModelNet40Ply2048DataModule) == (9840, 2468)
    assert dataset_lens(ScanObjectNNDataModule) == (11416, 2882)
    assert dataset_lens(ShapeNet55DataModule) == (41952, 10518)
    assert dataset_lens(ShapeNet55DataModule, in_memory=True) == (41952, 10518)
    assert dataset_lens(ShapeNetPartDataModule) == (13998, 2874)

    for way in (5, 10):
        for shot in (10, 20):
            for fold in range(10):
                assert dataset_lens(
                    ModelNet40FewShotDataModule, way=way, shot=shot, fold=fold
                ) == (way * shot, way * 20)

    print("All checks passed!")
