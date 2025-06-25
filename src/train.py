import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from dataset import PotholeDataset, load_dataset
from model import HybridConvTransformer
import torchvision.transforms as transforms
import os
import numpy as np
from collections import Counter

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data
data_dir = "C:\\Users\\prasa\\OneDrive\\Desktop\\miniv\\dataset"
X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(data_dir)
train_dataset = PotholeDataset(X_train, y_train, train_transform)
val_dataset = PotholeDataset(X_val, y_val, val_transform)
test_dataset = PotholeDataset(X_test, y_test, val_transform)

# Check class balance
train_labels = [label.item() for _, label in train_dataset]
class_counts = Counter(train_labels)
print("Class distribution:", class_counts)
# Weighted sampler for imbalance
weights = [1.0 / class_counts[label] for label in train_labels]
sampler = WeightedRandomSampler(weights, len(train_labels), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridConvTransformer().to(device)
# Freeze conv layers if using pretrained weights
# for param in model.conv.parameters():
#     param.requires_grad = False
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))  # Adjust based on imbalance
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# Training
def train_model(epochs=30, patience=5):
    best_val_acc = 0
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_correct, train_total = 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            if images is None:
                continue
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

        # Validation
        model.eval()
        correct, total = 0, 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if images is None:
                    continue
                images, labels = images.to(device), labels.to(device)
                labels = labels.unsqueeze(1)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total if total > 0 else 0
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/pothole_hybrid_model.pth")
            print("Saved best model")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
        scheduler.step(val_acc)

    # Test
    model.load_state_dict(torch.load("models/pothole_hybrid_model.pth"))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            if images is None:
                continue
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total if total > 0 else 0
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_model(epochs=30)