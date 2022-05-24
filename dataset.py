import os
from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, Subset
from torchvision import transforms as transforms
from torchvision.io import read_image


class TrafficSignDataset(Dataset):
    # ID to className dict was taken from https://www.kaggle.com/code/shivank856/gtsrb-cnn-98-test-accuracy,
    # by SHIVANK SHARMA
    classes = {
        0: "Speed limit (20km/h)",
        1: "Speed limit (30km/h)",
        2: "Speed limit (50km/h)",
        3: "Speed limit (60km/h)",
        4: "Speed limit (70km/h)",
        5: "Speed limit (80km/h)",
        6: "End of speed limit (80km/h)",
        7: "Speed limit (100km/h)",
        8: "Speed limit (120km/h)",
        9: "No passing",
        10: "No passing veh over 3.5 tons",
        11: "Right-of-way at intersection",
        12: "Priority road",
        13: "Yield",
        14: "Stop",
        15: "No vehicles",
        16: "Veh > 3.5 tons prohibited",
        17: "No entry",
        18: "General caution",
        19: "Dangerous curve left",
        20: "Dangerous curve right",
        21: "Double curve",
        22: "Bumpy road",
        23: "Slippery road",
        24: "Road narrows on the right",
        25: "Road work",
        26: "Traffic signals",
        27: "Pedestrians",
        28: "Children crossing",
        29: "Bicycles crossing",
        30: "Beware of ice/snow",
        31: "Wild animals crossing",
        32: "End speed + passing limits",
        33: "Turn right ahead",
        34: "Turn left ahead",
        35: "Ahead only",
        36: "Go straight or right",
        37: "Go straight or left",
        38: "Keep right",
        39: "Keep left",
        40: "Roundabout mandatory",
        41: "End of no passing",
        42: "End no passing veh > 3.5 tons",
    }

    def __init__(self, data_root: str, split: str, transform: transforms.Compose, crop: bool = True):
        self.data_root = data_root
        self.split = split
        self.df = pd.read_csv(os.path.join(data_root, split + ".csv"))
        self.transform = transform
        self.crop = crop

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.df.iloc[idx]["Path"]
        img = read_image(os.path.join(self.data_root, img_path))
        if self.crop:
            # crop the image to the size of the traffic sign
            xmin = self.df.iloc[idx]["Roi.X1"]
            ymin = self.df.iloc[idx]["Roi.Y1"]
            xmax = self.df.iloc[idx]["Roi.X2"]
            ymax = self.df.iloc[idx]["Roi.Y2"]
            img = transforms.functional.crop(img, ymin, xmin, ymax - ymin, xmax - xmin)

        if self.transform:
            img = self.transform(img)

        class_id = self.df.iloc[idx]["ClassId"]
        class_id = torch.tensor(class_id)

        return img, class_id


def compute_mean_and_std(dataset: Union[Dataset, Subset]):
    mean = 0.0
    std = 0.0
    nb_samples = 0.0
    for img, _ in dataset:
        mean += img.mean(axis=(1, 2))
        std += img.std(axis=(1, 2))
        nb_samples += 1.0
    mean /= nb_samples
    std /= nb_samples
    return mean, std


def plot_class_dist(
        dataset: Union[TrafficSignDataset, Subset],
        name: str,
        data_root: str,
        normalize: bool = False,
        weights: Optional[np.ndarray] = None,
) -> None:
    """
    Plots the class distribution of the dataset.
    """
    if isinstance(dataset, Subset):
        dataset = dataset.dataset.df.iloc[dataset.indices]
    elif isinstance(dataset, TrafficSignDataset):
        dataset = dataset.df
    class_dist = dataset.groupby("ClassId").size()
    class_dist = class_dist.rename(TrafficSignDataset.classes)
    class_dist = class_dist.sort_values(ascending=True)
    if normalize:
        class_dist = class_dist / class_dist.sum()
    plt.figure(figsize=(20, 10))
    plt.bar(class_dist.index, class_dist.values)
    plt.title(f"Class distribution of {name} set")
    plt.ylabel("Proportion")
    plt.xticks(rotation=90, fontsize=12)
    if weights:
        ax = plt.twinx()
        ax.plot(weights, color="red")
        ax.set_ylabel("Weight")
    plt.tight_layout()
    plt.plot()
    if weights:
        plt.savefig(os.path.join(data_root, f"class_dist_{name}_with_weights.png"))
    else:
        plt.savefig(os.path.join(data_root, f"class_dist_{name}.png"))


def enet_weighting(
        dataset: Union[TrafficSignDataset, Subset],
        c: float = 1.02,
        sort_first: bool = False,
) -> np.array:
    """
    Plots the class distribution of the dataset.
    """
    if isinstance(dataset, Subset):
        dataset = dataset.dataset.df.iloc[dataset.indices]
    elif isinstance(dataset, TrafficSignDataset):
        dataset = dataset.df
    class_dist = dataset.groupby("ClassId").size()
    if sort_first:
        class_dist = class_dist.sort_values(ascending=True)
    class_dist = class_dist / class_dist.sum()
    class_dist = 1 / np.log(c + class_dist)
    class_dist = class_dist.to_numpy()

    return class_dist

if __name__ == "__main__":
    from torch.utils.data import random_split
    import yaml

    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = TrafficSignDataset("data", "Train", transform)
    test_dataset = TrafficSignDataset("data", "Test", transform)
    dataset_size = len(train_dataset) + len(test_dataset)
    val_dataset_size = int(0.05 * dataset_size)
    train_dataset_size = len(train_dataset) - val_dataset_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_dataset_size, val_dataset_size],
        generator=torch.Generator().manual_seed(cfg["SEED"]),
    )

    # print("Computing mean and std of the train set...")
    # mean, std = compute_mean_and_std(train_dataset)
    # print(mean, std)

    plot_class_dist(train_dataset, "train", cfg["DATA_ROOT"])
    # plot_class_dist(val_dataset, "val", cfg["DATA_ROOT"])
    # plot_class_dist(test_dataset, "test", cfg["DATA_ROOT"])
    #
    # weights = enet_weighting(train_dataset)
    # plot_class_dist(train_dataset, "train", cfg["DATA_ROOT"], weights=weights)
