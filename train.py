import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from insightface.model_zoo import get_model
from PIL import Image
import numpy as np

# カスタムデータセット
class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# データ準備
def prepare_data(data_dir):
    image_paths = []
    labels = []
    class_map = {}

    for idx, person_dir in enumerate(os.listdir(data_dir)):
        class_map[idx] = person_dir
        person_path = os.path.join(data_dir, person_dir)
        if os.path.isdir(person_path):
            for img_file in os.listdir(person_path):
                if img_file.endswith(('jpg', 'png', 'jpeg')):
                    image_paths.append(os.path.join(person_path, img_file))
                    labels.append(idx)

    return image_paths, labels, class_map

# ArcFaceモデルのロードとカスタマイズ
def get_model_for_finetuning(num_classes):
    model = get_model("arcface_r50_v1")  # 事前学習済みArcFaceモデル
    model.eval()  # 評価モード
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # 出力層を調整
    return model

# 学習
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 検証
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

    print("Training finished.")
    return model

# データ準備
data_dir = "Bus_Occupant_Detect\\10_person_dataset"  # 10人分のデータを格納したディレクトリ
image_paths, labels, class_map = prepare_data(data_dir)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

# データ拡張とデータローダー
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ArcFaceモデルの正規化
])

train_dataset = FaceDataset(train_paths, train_labels, transform=transform)
val_dataset = FaceDataset(val_paths, val_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# モデルの準備
num_classes = len(set(labels))
model = get_model_for_finetuning(num_classes)
model = model.cuda()

# 再学習
finetuned_model = train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.0001)

# モデルの保存
torch.save(finetuned_model.state_dict(), "finetuned_arcface_model.pth")
