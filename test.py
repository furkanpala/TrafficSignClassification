import os

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    top_k_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader

from dataset import TrafficSignDataset
from model import VGG16
from utils import get_logger, create_dir_if_not_exists


def main():
    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    model_name = cfg["MODEL_NAME"]
    create_dir_if_not_exists(os.path.join("models", model_name, "logs"))
    create_dir_if_not_exists(os.path.join("models", model_name, "plots"))

    logger = get_logger(__name__, model_name)
    logger.info(cfg)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((cfg["RESIZE_HEIGHT"], cfg["RESIZE_WIDTH"])),
            transforms.ToTensor(),
            transforms.Normalize(cfg["MEAN"], cfg["STD"]),
        ]
    )
    test_dataset = TrafficSignDataset(cfg["DATA_ROOT"], "Test", transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True)
    model = VGG16(num_classes=cfg["NUM_CLASSES"])

    logger.info("Testing started")
    logger.info(f"Testing dataset size: {len(test_dataset)}")

    weights_path = os.path.join("models", model_name, "weights", "min_val_loss.pt")
    logger.info(f"Loading weights from {weights_path}")
    weights = torch.load(weights_path, map_location="cpu")
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    model = model.to(device)

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

    cr = classification_report(
        gt_ids_all, pred_ids_all, target_names=TrafficSignDataset.classes.values()
    )
    logger.info(cr)

    cm = confusion_matrix(gt_ids_all, pred_ids_all, normalize=None)
    _, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(
        cm, display_labels=TrafficSignDataset.classes.values()
    )
    disp.plot(include_values=True, ax=ax, xticks_rotation="vertical")
    plt.savefig(os.path.join("models", model_name, "plots", "cm.png"))

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
        img, label = test_dataset[sample_idx]
        logits = model(img.unsqueeze(0).to(device)).squeeze().cpu().detach()
        probs = torch.softmax(logits, dim=0)
        pred_label = torch.argmax(probs)
        figure.add_subplot(rows, cols, i)
        plt.tight_layout()
        plt.xlabel(
            "GT: {}\nPred: {}\nConf: {:.2f}".format(
                TrafficSignDataset.classes[label.item()],
                TrafficSignDataset.classes[pred_label.item()],
                probs[pred_label].item(),
            )
        )
        plt.xticks([])
        plt.yticks([])
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img * np.array(cfg["STD"]) + np.array(cfg["MEAN"])
        img *= 255
        plt.imshow(img.astype(np.uint8))

    plt.savefig(os.path.join("models", model_name, "plots", "test_samples.png"))


if __name__ == "__main__":
    main()
