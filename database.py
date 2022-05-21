import os

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
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

    def __init__(self, data_root, split, transform, crop=True):
        self.data_root = data_root
        self.split = split
        self.dataset = pd.read_csv(os.path.join(data_root, split + ".csv"))
        self.transform = transform
        self.crop = crop

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset.iloc[idx]["Path"]
        img = read_image(os.path.join(self.data_root, img_path))
        if self.crop:
            # crop the image to the size of the traffic sign
            xmin = self.dataset.iloc[idx]["Roi.X1"]
            ymin = self.dataset.iloc[idx]["Roi.Y1"]
            xmax = self.dataset.iloc[idx]["Roi.X2"]
            ymax = self.dataset.iloc[idx]["Roi.Y2"]
            img = transforms.functional.crop(img, ymin, xmin, ymax - ymin, xmax - xmin)

        if self.transform:
            img = self.transform(img)

        class_id = self.dataset.iloc[idx]["ClassId"]
        class_id = torch.tensor(class_id)

        return img, class_id

    def compute_mean_and_std(self):
        mean = 0.0
        std = 0.0
        nb_samples = 0.0
        for i in range(len(self)):
            img = self.__getitem__(i)[0]
            mean += img.mean(axis=(1, 2))
            std += img.std(axis=(1, 2))
            nb_samples += 1.0
        mean /= nb_samples
        std /= nb_samples
        return mean, std

    def get_stats(self):
        # Plot histogram of class distribution
        class_dist = self.dataset.groupby("ClassId").size()
        class_dist = class_dist.rename(TrafficSignDataset.classes)
        class_dist = class_dist.sort_values(ascending=True)
        class_dist = class_dist / class_dist.sum()
        plt.figure(figsize=(20, 10))
        plt.bar(class_dist.index, class_dist.values)
        plt.title(f"Class distribution of {self.split} set")
        plt.ylabel("Proportion")
        plt.xticks(rotation=90, fontsize=12)
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(self.data_root, f"class_dist_{self.split}.png"))
