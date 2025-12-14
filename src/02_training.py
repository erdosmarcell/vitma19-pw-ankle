import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

class CSVImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.label_map = {
            label: idx for idx, label in enumerate(sorted(self.df["label"].unique()))
        }

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
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, name, best_path=None):
    best_val_acc = 0.0
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        epoch_loss = running_loss / total
        loss_history.append(epoch_loss)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(
            f"[{name}] Epoch {epoch:02d}/{epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"Best Val Acc ({name}): {best_val_acc:.4f}")
    return loss_history, best_val_acc

def main():
    print("\n=== Hyperparameters ===")
    print(f"Base model (TinyNet):")
    print(f"  Epochs: {config.EPOCHS_base}")
    print(f"  Batch size: {config.BATCH_SIZE_base}")
    print(f"  Learning rate: {config.LEARNING_RATE_base}")

    print(f"\nResNet18:")
    print(f"  Epochs: {config.EPOCHS_resnet18}")
    print(f"  Batch size: {config.BATCH_SIZE_resnet18}")
    print(f"  Learning rate: {config.LEARNING_RATE_resnet18}")

    TRAIN_CSV = config.TRAIN_CSV
    VAL_CSV   = config.VAL_CSV
    #TEST_CSV  = "/data/split/test.csv"

    print("=== 02_training.py - Baseline vs Pre-trained ===")

    transform_tiny_train = transforms.Compose([
        transforms.Resize(140),
        transforms.CenterCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    transform_tiny_eval = transforms.Compose([
        transforms.Resize(140),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    transform_resnet_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    transform_resnet_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    train_ds_tiny = CSVImageDataset(TRAIN_CSV, transform_tiny_train)
    val_ds_tiny   = CSVImageDataset(VAL_CSV,   transform_tiny_eval)

    train_ds_rn = CSVImageDataset(TRAIN_CSV, transform_resnet_train)
    val_ds_rn   = CSVImageDataset(VAL_CSV,   transform_resnet_eval)

    num_classes = len(train_ds_tiny.label_map)

    train_loader_tiny = DataLoader(
        train_ds_tiny, batch_size=config.BATCH_SIZE_base, shuffle=True, num_workers=4
    )
    val_loader_tiny = DataLoader(
        val_ds_tiny, batch_size=config.BATCH_SIZE_base, shuffle=False, num_workers=4
    )

    train_loader_rn = DataLoader(
        train_ds_rn, batch_size=config.BATCH_SIZE_resnet18, shuffle=True, num_workers=2
    )
    val_loader_rn = DataLoader(
        val_ds_rn, batch_size=config.BATCH_SIZE_resnet18, shuffle=False, num_workers=2
    )

    criterion = nn.CrossEntropyLoss()

    print("\n=== Training ResNet18 (pre-trained) ===")
    resnet = build_resnet18(num_classes).to(DEVICE)
    
    print("\n=== ResNet18 architecture ===")
    print(resnet)

    opt_rn = optim.Adam(resnet.fc.parameters(), lr=config.LEARNING_RATE_resnet18)

    train_model(
        model=resnet,
        train_loader=train_loader_rn,
        val_loader=val_loader_rn,
        criterion=criterion,
        optimizer=opt_rn,
        epochs=config.EPOCHS_resnet18,
        name="resnet18",
        best_path=config.RES_BEST_PATH
    )

    torch.save(resnet.state_dict(), config.RES_FINAL_PATH)

    print("\n=== Training TinyNet (baseline) ===")
    tiny = TinyNet(num_classes).to(DEVICE)
    print("\n=== TinyNet architecture ===")
    print(tiny)
    
    opt_tiny = optim.Adam(tiny.parameters(), lr=config.LEARNING_RATE_base)

    train_model(
        model=tiny,
        train_loader=train_loader_tiny,
        val_loader=val_loader_tiny,
        criterion=criterion,
        optimizer=opt_tiny,
        epochs=config.EPOCHS_base,
        name="tinynet",
        best_path=config.BASE_BEST_PATH
    )

    torch.save(tiny.state_dict(), config.BASE_FINAL_PATH)

if __name__ == "__main__":
    main()
