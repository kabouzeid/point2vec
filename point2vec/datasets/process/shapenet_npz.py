"""
builds the npz dataset files from the original dataset
"""
if __name__ == "__main__":
    import os

    import numpy as np
    from tqdm import tqdm

    from point2vec.datasets import ShapeNet55

    root = "./data/ShapeNet55"

    def save(split: str):
        dataset = ShapeNet55(root, split=split)
        data_list = []
        label_list = []
        for idx in tqdm(range(len(dataset)), f"Loading ShapeNet55 {split} split"):
            data, label = dataset[idx]
            data_list.append(data)
            label_list.append(label)
        data = np.stack(data_list)
        labels = np.stack(label_list)
        save_path = os.path.join(root, f"shapenet_{split}.npz")
        print(f"Saving to {save_path}")
        np.savez(save_path, data=data, labels=labels)

    save("train")
    save("test")
