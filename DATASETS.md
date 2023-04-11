# Datasets

All datasets go into a `data` directory at the root of the project.
You can use for example `mkdir data` or `ln -s /my/data/storage data` to create this directory.

In the end the overall directory structure should be as follows (generated with `tree data -L 2`).
```
data
├── modelnet40_ply_hdf5_2048
│   ├── ply_data_test0.h5
│   ├── ply_data_test_0_id2file.json
│   ├── ply_data_test1.h5
│   ├── ply_data_test_1_id2file.json
│   ├── ply_data_train0.h5
│   ├── ply_data_train_0_id2file.json
│   ├── ply_data_train1.h5
│   ├── ply_data_train_1_id2file.json
│   ├── ply_data_train2.h5
│   ├── ply_data_train_2_id2file.json
│   ├── ply_data_train3.h5
│   ├── ply_data_train_3_id2file.json
│   ├── ply_data_train4.h5
│   ├── ply_data_train_4_id2file.json
│   ├── shape_names.txt
│   ├── test_files.txt
│   └── train_files.txt
├── ModelNetFewshot
│   ├── 10way_10shot
│   ├── 10way_20shot
│   ├── 5way_10shot
│   └── 5way_20shot
├── ScanObjectNN
│   ├── main_split
│   ├── main_split_nobg
│   ├── split1
│   ├── split1_nobg
│   ├── split2
│   ├── split2_nobg
│   ├── split3
│   ├── split3_nobg
│   ├── split4
│   └── split4_nobg
├── ShapeNet55
│   ├── shapenet_pc
│   ├── shapenet_test.npz
│   ├── shapenet_train.npz
│   ├── test.txt
│   ├── train_25.txt
│   ├── train_50.txt
│   ├── train_75.txt
│   └── train.txt
└── shapenetcore_partanno_segmentation_benchmark_v0_normal
    ├── 02691156
    ├── 02773838
    ├── 02954340
    ├── 02958343
    ├── 03001627
    ├── 03261776
    ├── 03467517
    ├── 03624134
    ├── 03636649
    ├── 03642806
    ├── 03790512
    ├── 03797390
    ├── 03948459
    ├── 04099429
    ├── 04225987
    ├── 04379243
    ├── synsetoffset2category.txt
    └── train_test_split
```

## ShapeNet

Download `ShapeNet55.zip` from [Google Drive](https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK/view?usp=sharing) into the `data` directory (see also [Point-BERT DATASET.md](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md)).
Then
```bash
cd data
unzip ShapeNet55.zip
cp ../.metadata/ShapeNet55/* ShapeNet55/ # train.txt and test.txt split files
cd ..
python -m point2vec.datasets.process.shapenet_npz
```
The last step saves the whole dataset in `shapenet_train.npz` and `shapenet_test.npz` files which are used by the `--data.in_memory true` flag.
This is recommended if you have slow disk I/O, and necessary if you want to reproduce the exact results from our paper.

## ScanObjectNN

Download `h5_files.zip` from [the official Website](https://hkust-vgd.github.io/scanobjectnn) into the `data` directory by agreeing to the Terms of Use.
Then
```bash
cd data
unzip h5_files.zip
mv h5_files ScanObjectNN
```

## ModelNet40

```bash
cd data
wget --no-check-certificate https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
unzip modelnet40_ply_hdf5_2048.zip
```

## ModelNet Few Shot

Download the directory from [Google Drive](https://drive.google.com/drive/folders/1gqvidcQsvdxP_3MdUr424Vkyjb_gt7TW?usp=sharing) into the `data` directory (see also [Point-BERT DATASET.md](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md)).

## ShapeNetPart

```bash
cd data
wget --no-check-certificate https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
```
