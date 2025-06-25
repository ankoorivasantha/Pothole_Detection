import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

class PotholeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image not found at {self.image_paths[idx]}")
            return None, None
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, label

def load_dataset(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    image_paths, labels = [], []
    for label_dir in ['potholes', 'normal']:  # Changed 'pothole' to 'potholes'
        dir_path = os.path.join(data_dir, label_dir)
        if not os.path.exists(dir_path):
            print(f"Warning: {dir_path} not found")
            continue
        for img_name in os.listdir(dir_path):
            if img_name.lower().endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(dir_path, img_name))
                labels.append(1 if label_dir == 'potholes' else 0)
    if not image_paths:
        raise ValueError("No valid images found")
    # Split: 70% train, 20% val, 10% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=0.1, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2222, random_state=42  # 0.2222 ~ 20/90
    )
    return X_train, X_val, X_test, y_train, y_val, y_test