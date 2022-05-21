import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import (
    f1_score,
)
from torch.utils.data import DataLoader, random_split

from database import TrafficSignDataset
from model import VGG16
from utils import create_dir_if_not_exists, get_logger


def main():
    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    model_name = cfg["MODEL_NAME"]

    create_dir_if_not_exists(os.path.join("models", model_name))
    create_dir_if_not_exists(os.path.join("models", model_name, "weights"))
    create_dir_if_not_exists(os.path.join("models", model_name, "logs"))
    create_dir_if_not_exists(os.path.join("models", model_name, "plots"))

    logger = get_logger(__name__, model_name)
    logger.info(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    mean, std = cfg["MEAN"], cfg["STD"]
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = TrafficSignDataset("data", "Train", transform)
    test_dataset = TrafficSignDataset("data", "Test", transform)
    test_dataset, val_dataset = random_split(
        test_dataset,
        [len(test_dataset) // 2, len(test_dataset) // 2],
        generator=torch.Generator().manual_seed(cfg["SEED"]),
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True)

    model = VGG16(num_classes=43)

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
                model.module.state_dict(),
                os.path.join("models", model_name, "weights", "min_val_loss.pt"),
            )

        if micro_f1 > max_micro_f1_score:
            max_micro_f1_score = micro_f1
            logger.info("Max micro f1 score found, saving model...")
            torch.save(
                model.module.state_dict(),
                os.path.join("models", model_name, "weights", "max_micro_f1.pt"),
            )

        if macro_f1 > max_macro_f1_score:
            max_macro_f1_score = macro_f1
            logger.info("Max macro f1 score found, saving model...")
            torch.save(
                model.module.state_dict(),
                os.path.join("models", model_name, "weights", "max_macro_f1.pt"),
            )

        if epoch % cfg["SAVE_EVERY_EPOCH"] == 0:
            logger.info("Saving model...")
            torch.save(
                model.module.state_dict(),
                os.path.join("models", model_name, "weights", f"epoch{epoch}.pt"),
            )

        mean_train_losses.append(mean_train_loss)
        mean_val_losses.append(mean_val_loss)
        micro_f1_scores.append(micro_f1)
        macro_f1_scores.append(macro_f1)
        lr_values.append(current_lr)

        with open(os.path.join("models", model_name, "plots", "data.pkl"), "wb") as f:
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

    with open(os.path.join("models", model_name, "plots", "data.pkl"), "rb") as f:
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
        plt.savefig(os.path.join("models", model_name, "plots", "loss.png"))

        plt.figure()
        plt.plot(micro_f1_scores, label="Micro F1")
        plt.plot(macro_f1_scores, label="Macro F1")
        plt.xlabel("Epoch")
        plt.title("F1 Scores over validation set")
        plt.legend()
        plt.savefig(os.path.join("models", model_name, "plots", "f1_scores.png"))

        plt.figure()
        plt.plot(lr_values)
        plt.xlabel("Epoch")
        plt.title("Learning rate decay")
        plt.savefig(os.path.join("models", model_name, "plots", "lr.png"))


if __name__ == "__main__":
    main()
