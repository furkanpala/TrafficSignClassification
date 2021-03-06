from flask import Flask, request, jsonify
from flask import app, render_template, redirect
import glob
from PIL import Image
import io
import numpy as np
from torchvision.transforms import transforms
import yaml
import os
import torch

from model import VGG16
from dataset import TrafficSignDataset
import pandas as pd

app = Flask(__name__)
selected_model = None

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

model = VGG16(num_classes=cfg["NUM_CLASSES"], init_weights=False, config="D")
model = model.eval()

train_csv_path = os.path.join(cfg["DATA_ROOT"], "Train.csv")
test_csv_path = os.path.join(cfg["DATA_ROOT"], "Test.csv")

train_df = None
test_df = None

if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)

if os.path.exists(test_csv_path):
    test_df = pd.read_csv(test_csv_path)


def list_models(selected_model):
    models = [
        (path, selected_model == path)
        for path in glob.glob("models/**/*.pt", recursive=True)
        + glob.glob("models/**/*.pth", recursive=True)
    ]
    models = [(None, selected_model is None)] + models
    return models


@app.route("/")
def index():
    global selected_model
    models = list_models(selected_model)
    return render_template("index.html", models=models)


@app.route("/select_model", methods=["POST"])
def select_model():
    global selected_model
    selected_model = request.form.get("model_selection_list", None)
    if os.path.exists(selected_model):
        weights = torch.load(selected_model, map_location="cpu")
        weights = {k.replace("module.", ""): v for k, v in weights.items()}
        model.load_state_dict(weights)

    return redirect("/")


@app.route("/predict", methods=["POST"])
def predict():
    img = request.files["image"]
    filename = img.filename
    np_img = np.frombuffer(img.read(), np.uint8)
    img = Image.open(io.BytesIO(np_img)).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((cfg["RESIZE_HEIGHT"], cfg["RESIZE_WIDTH"])),
            transforms.ToTensor(),
            transforms.Normalize(cfg["MEAN"], cfg["STD"]),
        ]
    )

    img = transform(img).unsqueeze(0)
    logits = model(img).squeeze().cpu().detach()
    probs = torch.softmax(logits, dim=0)
    topk_probs, topk_indices = torch.topk(probs, k=5)
    topk_probs = topk_probs.tolist()
    topk_probs = [round(p, 6) for p in topk_probs]
    topk_indices = topk_indices.tolist()
    topk_labels = [TrafficSignDataset.classes[ind] for ind in topk_indices]

    class_id = None
    if isinstance(test_df, pd.DataFrame):
        row = test_df.loc[test_df["Path"].str.contains(filename)]
        if len(row) > 0:
            class_id = row["ClassId"].item()
    elif class_id is None and isinstance(train_df, pd.DataFrame):
        row = train_df.loc[train_df["Path"].str.contains(filename)]
        if len(row) > 0:
            class_id = row["ClassId"].item()

    return jsonify(
        {
            "probs": topk_probs,
            "labels": topk_labels,
            "gt_label": TrafficSignDataset.classes.get(class_id, "Unknown"),
        }
    )


if __name__ == "__main__":
    app.run()
