import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

class SampleDataset(Dataset):
    def __init__(self, img_dir, json_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []

        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            for item in json_data:
                img_file = item.get("file_upload") or ""
                label = None
                annotations = item.get("annotations", [])
                if annotations:
                    result_list = annotations[0].get("result", [])
                    if result_list:
                        choices = result_list[0].get("value", {}).get("choices", [])
                        if choices:
                            label = choices[0].split("_", 1)[-1]
                if img_file:
                    if "-" in img_file:
                        img_file = img_file.split("-", 1)[-1]
                    self.data.append({"img_file": img_file, "label": label})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file = self.data[idx]["img_file"]
        label = self.data[idx]["label"]
        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_file

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
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def run_inference(model, loader, label_map):
    model.eval()
    results = []
    with torch.no_grad():
        for images, true_labels, img_files in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for img_file, true_label, pred_idx in zip(img_files, true_labels, preds):
                pred_label = list(label_map.keys())[pred_idx]
                results.append({"image": img_file, "true_label": true_label, "pred_label": pred_label})
    return results

def main():
    SAMPLE_DIR = "/data/anklealign/sample"
    json_files = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(".json")]
    if len(json_files) != 1:
        raise RuntimeError(f"Sample dir should contain exactly one JSON file, found: {len(json_files)}")
    JSON_PATH = os.path.join(SAMPLE_DIR, json_files[0])
    print(f"Using JSON file to check inference: {JSON_PATH}")

    transform = transforms.Compose([
        transforms.Resize(140),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    dataset = SampleDataset(SAMPLE_DIR, JSON_PATH, transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    all_labels = ['Neutralis', 'Pronacio', 'Szupinacio']
    label_map = {label: idx for idx, label in enumerate(all_labels)}
    num_classes = len(label_map)
    print("Classes:", all_labels)

    tinynet = TinyNet(num_classes).to(DEVICE)
    tinynet.load_state_dict(torch.load("/data/tinynet_best.pth", map_location=DEVICE))

    resnet18 = build_resnet18(num_classes).to(DEVICE)
    resnet18.load_state_dict(torch.load("/data/resnet18_best.pth", map_location=DEVICE))

    print("\nRunning inference with TinyNet...")
    results_tiny = run_inference(tinynet, loader, label_map)

    print("\nRunning inference with ResNet18...")
    results_rn = run_inference(resnet18, loader, label_map)

    print("\n=== Comparison of model predictions ===")
    for tiny_res, rn_res in zip(results_tiny, results_rn):
        print(f"{tiny_res['image']}: True={tiny_res['true_label']} | "
              f"TinyNet={tiny_res['pred_label']} | ResNet18={rn_res['pred_label']}")

if __name__ == "__main__":
    main()
