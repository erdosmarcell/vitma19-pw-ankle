import os
import pandas as pd
import config
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

class CSVImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)

        self.df = self.df[self.df["label"].notna()]
        self.df = self.df[self.df["label"] != ""].reset_index(drop=True)

        self.transform = transform

        self.labels = ["Neutralis", "Pronacio", "Szupinacio"]
        self.label_map = {label: idx for idx, label in enumerate(self.labels)}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = self.label_map[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label

class TinyNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def evaluate_model(model, loader, label_names, model_name):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== {model_name} RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=label_names
    ))

    return acc, f1

def main():
    CONSENSUS_TEST_CSV = config.CONSENSUS_TEST_CSV

    transform_tiny = transforms.Compose([
        transforms.Resize(140),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    transform_resnet = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    ds_tiny = CSVImageDataset(CONSENSUS_TEST_CSV, transform_tiny)
    ds_rn = CSVImageDataset(CONSENSUS_TEST_CSV, transform_resnet)

    loader_tiny = DataLoader(ds_tiny, batch_size=config.BATCH_SIZE_base, shuffle=False)
    loader_rn = DataLoader(ds_rn, batch_size=config.BATCH_SIZE_resnet18, shuffle=False)

    num_classes = len(ds_tiny.label_map)
    label_names = ds_tiny.labels

    print(f"\nEvaluating on {len(ds_tiny)} consensus-labeled images")
    print(f"Classes: {label_names}")

    tinynet = TinyNet(num_classes).to(DEVICE)
    tinynet.load_state_dict(torch.load(config.BASE_BEST_PATH, map_location=DEVICE))

    resnet = build_resnet18(num_classes).to(DEVICE)
    resnet.load_state_dict(torch.load(config.RES_BEST_PATH, map_location=DEVICE))

    acc_tiny, f1_tiny = evaluate_model(
        tinynet, loader_tiny, label_names, "TinyNet (baseline)"
    )

    acc_rn, f1_rn = evaluate_model(
        resnet, loader_rn, label_names, "ResNet18 (pre-trained)"
    )

    print("\n=== MODEL COMPARISON SUMMARY ===")
    print(f"TinyNet  | Accuracy: {acc_tiny:.4f} | Macro F1: {f1_tiny:.4f}")
    print(f"ResNet18 | Accuracy: {acc_rn:.4f} | Macro F1: {f1_rn:.4f}")

    diff_acc = acc_rn - acc_tiny
    diff_f1  = f1_rn - f1_tiny

    print("\nDifference (ResNet18 - TinyNet):")
    print(f"Accuracy diff: {diff_acc:+.4f}")
    print(f"Macro F1 diff: {diff_f1:+.4f}")


if __name__ == "__main__":
    main()
