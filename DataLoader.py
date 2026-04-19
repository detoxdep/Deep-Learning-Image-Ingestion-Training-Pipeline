import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Transform setup
data_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Define the paths
data_dir = './data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# 1. Train Loader
train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2. Val Loader
val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. Test Loader
test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"DataLoader Ready!")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")