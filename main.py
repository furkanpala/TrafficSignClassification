import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image


# TODO:
# Loss weights ???
#   Plot class vs support
#   Class imbalance ?
#   E-Net weighting, maybe

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


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_logger(model_name):
    path = os.path.join("models", model_name, "logs", "log.log")
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s\n%(message)s")

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


class VGG(nn.Module):
    def __init__(self, init_weights=True, num_classes=43):
        super(VGG, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


class TrafficSignDataset(Dataset):
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
        class_dist = class_dist.rename(classes)
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


def main():
    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    MODEL_NAME = cfg["MODEL_NAME"]

    create_dir_if_not_exists(os.path.join("models", MODEL_NAME))
    create_dir_if_not_exists(os.path.join("models", MODEL_NAME, "weights"))
    create_dir_if_not_exists(os.path.join("models", MODEL_NAME, "logs"))
    create_dir_if_not_exists(os.path.join("models", MODEL_NAME, "plots"))

    logger = get_logger(MODEL_NAME)
    logger.info(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    mean, std = (0.3589, 0.3177, 0.3357), (0.1457, 0.1530, 0.1624)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = TrafficSignDataset("data", "Train", transform)
    # train_dataset.get_stats()
    # mean, std = train_dataset.compute_mean_and_std()
    # print(mean, std)

    test_dataset = TrafficSignDataset("data", "Test", transform)
    test_dataset, val_dataset = random_split(
        test_dataset,
        [len(test_dataset) // 2, len(test_dataset) // 2],
        generator=torch.Generator().manual_seed(cfg["SEED"]),
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True)

    model = VGG(num_classes=43)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["LR"],
        momentum=cfg["MOMENTUM"],
        weight_decay=cfg["WD"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    mean_train_losses = []
    mean_val_losses = []
    micro_f1_scores = []
    macro_f1_scores = []
    lr_values = []
    min_val_loss = float("inf")
    max_micro_f1_score = 0.0
    max_macro_f1_score = 0.0
    for epoch in range(1, cfg["EPOCHS"] + 1):
        mean_train_loss = 0.0
        mean_val_loss = 0.0
        gt_ids_all = []
        pred_ids_all = []

        # Training loop
        for img, gt_ids in train_loader:
            img = img.to(device)
            gt_ids = gt_ids.to(device)

            optimizer.zero_grad()

            logits = model(img)

            loss = loss_fn(logits, gt_ids)
            mean_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        # Validation loop
        for img, gt_ids in val_loader:
            img = img.to(device)
            gt_ids = gt_ids.to(device)

            logits = model(img)
            loss = loss_fn(logits, gt_ids)
            mean_val_loss += loss.item()

            pred_ids = torch.argmax(logits, dim=1)

            gt_ids_all.extend(gt_ids.cpu().detach().tolist())
            pred_ids_all.extend(pred_ids.cpu().detach().tolist())

        scheduler.step(mean_val_loss)

        mean_train_loss /= len(train_loader)
        mean_val_loss /= len(val_loader)
        micro_f1 = f1_score(gt_ids_all, pred_ids_all, average="micro")
        macro_f1 = f1_score(gt_ids_all, pred_ids_all, average="macro")
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch {}: train loss: {:.4f}, val loss: {:.4f}, micro_f1: {:.4f}, macro_f1: {:.4f}, lr: {}".format(
                epoch, mean_train_loss, mean_val_loss, micro_f1, macro_f1, current_lr
            )
        )

        if mean_val_loss < min_val_loss:
            min_val_loss = mean_val_loss
            logger.info("Min val loss found, saving model...")
            torch.save(
                model.state_dict(),
                os.path.join("models", MODEL_NAME, "weights", "min_val_loss.pt"),
            )

        if micro_f1 > max_micro_f1_score:
            max_micro_f1_score = micro_f1
            logger.info("Max micro f1 score found, saving model...")
            torch.save(
                model.state_dict(),
                os.path.join("models", MODEL_NAME, "weights", "max_micro_f1.pt"),
            )

        if macro_f1 > max_macro_f1_score:
            max_macro_f1_score = macro_f1
            logger.info("Max macro f1 score found, saving model...")
            torch.save(
                model.state_dict(),
                os.path.join("models", MODEL_NAME, "weights", "max_macro_f1.pt"),
            )

        if epoch % cfg["SAVE_EVERY_EPOCH"] == 0:
            logger.info("Saving model...")
            torch.save(
                model.state_dict(),
                os.path.join("models", MODEL_NAME, "weights", f"epoch{epoch}.pt"),
            )

        mean_train_losses.append(mean_train_loss)
        mean_val_losses.append(mean_val_loss)
        micro_f1_scores.append(micro_f1)
        macro_f1_scores.append(macro_f1)
        lr_values.append(current_lr)

        with open(os.path.join("models", MODEL_NAME, "plots", "data.pkl"), "wb") as f:
            pickle.dump(
                {
                    "train_losses": mean_train_losses,
                    "val_losses": mean_val_losses,
                    "micro_f1_scores": micro_f1_scores,
                    "macro_f1_scores": macro_f1_scores,
                    "lr": lr_values,
                },
                f,
            )

    """
        +----------------------------------------------------------------------------------------+
        |                                          TESTING                                       |
        +----------------------------------------------------------------------------------------+
    """
    logger.info("Testing started")

    weights_path = os.path.join("models", MODEL_NAME, "weights", "min_val_loss.pt")
    logger.info(f"Loading weights from {weights_path}")
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    gt_ids_all = []
    pred_ids_all = []
    logits_all = []
    for img, gt_ids in test_loader:
        img = img.to(device)
        gt_ids = gt_ids.to(device)

        logits = model(img)
        pred_ids = torch.argmax(logits, dim=1)

        gt_ids_all.extend(gt_ids.cpu().detach().tolist())
        pred_ids_all.extend(pred_ids.cpu().detach().tolist())
        logits_all.extend(logits.cpu().detach().tolist())

    micro_f1 = f1_score(gt_ids_all, pred_ids_all, average="micro")
    macro_f1 = f1_score(gt_ids_all, pred_ids_all, average="macro")
    acc = accuracy_score(gt_ids_all, pred_ids_all)
    top_5_acc = top_k_accuracy_score(gt_ids_all, logits_all)

    logger.info(
        "Testing results\nMicro F1 Score: {:.4f}\nMacro F1 Score: {:.4f}\nAccuracy: {:.4f}\nTop-5 Accuracy: {:.4f}".format(
            micro_f1, macro_f1, acc, top_5_acc
        )
    )

    cr = classification_report(gt_ids_all, pred_ids_all, target_names=classes.values())
    logger.info(cr)

    cm = confusion_matrix(gt_ids_all, pred_ids_all, normalize=None)
    _, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(include_values=False, ax=ax)
    plt.savefig(os.path.join("models", MODEL_NAME, "plots", "cm.png"))

    # df_cm = pd.DataFrame(cm, range(43), range(43))
    # plt.figure(figsize=(20, 20))
    # sb.heatmap(df_cm, annot=True, fmt=".2f")
    # plt.plot()
    # plt.savefig(os.path.join("models", MODEL_NAME, "plots", "cm.png"))

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
        img, label = test_dataset[sample_idx]
        logits = model(img.unsqueeze(0)).squeeze()
        probs = torch.softmax(logits, dim=0)
        pred_label = torch.argmax(probs)
        figure.add_subplot(rows, cols, i)
        plt.tight_layout()
        plt.xlabel(
            "GT: {}\nPred: {}\nConf: {:.2f}".format(
                classes[label.item()],
                classes[pred_label.item()],
                probs[pred_label].item(),
            )
        )
        plt.xticks([])
        plt.yticks([])
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img * np.array(std) + np.array(mean)
        img *= 255
        plt.imshow(img.astype(np.uint8))
    plt.show()
    plt.savefig(os.path.join("models", MODEL_NAME, "plots", "samples.png"))

    with open(os.path.join("models", MODEL_NAME, "plots", "data.pkl"), "rb") as f:
        data = pickle.load(f)
        train_losses = data["train_losses"]
        val_losses = data["val_losses"]
        micro_f1_scores = data["micro_f1_scores"]
        macro_f1_scores = data["macro_f1_scores"]
        lr_values = data["lr"]

        plt.figure()
        plt.plot(train_losses, label="Train loss")
        plt.plot(val_losses, label="Validation loss")
        plt.xlabel("Epoch")
        plt.title("Avg. training and validation losses")
        plt.legend()
        plt.savefig(os.path.join("models", MODEL_NAME, "plots", "loss.png"))

        plt.figure()
        plt.plot(micro_f1_scores, label="Micro F1")
        plt.plot(macro_f1_scores, label="Macro F1")
        plt.xlabel("Epoch")
        plt.title("F1 Scores over validation set")
        plt.legend()
        plt.savefig(os.path.join("models", MODEL_NAME, "plots", "f1_scores.png"))

        plt.figure()
        plt.plot(lr_values)
        plt.xlabel("Epoch")
        plt.title("Learning rate decay")
        plt.savefig(os.path.join("models", MODEL_NAME, "plots", "lr.png"))


if __name__ == "__main__":
    main()
