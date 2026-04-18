import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Define the "Recipe" for the images (Transforms)
data_transforms = transforms.Compose([
    transforms.Resize((150, 150)),  # Matches your model input
    transforms.ToTensor(),           # Converts pixels to 0-1 scale
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalized for RGB
])

# 2. Load the actual images
# Change 'data/train' to the actual path where the imagenes are saved
try:
    train_dataset = datasets.ImageFolder(root='./data/train', transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"Success! Found {len(train_dataset)} images.")
    print(f"Classes detected: {train_dataset.classes}")
    
except Exception as e:
    print(f"Error: Could not find the images. Check your folder path! \n{e}")

# 3. Create the "Loaders" (This feeds images in groups/batches)
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#print(f"Detected Classes: {train_dataset.classes}")
#print(f"Number of training images: {len(train_dataset)}")