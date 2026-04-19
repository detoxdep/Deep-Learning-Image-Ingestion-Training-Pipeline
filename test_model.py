import torch
from Models import ModelA
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2. Load the Test Data
data_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(root='./data/test', transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Load the Saved Brain
model = ModelA().to(device)
model.load_state_dict(torch.load('Model_A_weights.pth'))
model.eval()

# 4. Run the Test
correct = 0
total = 0

print("Evaluating on Test Set...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_acc = 100 * correct / total
print(f'\n--- FINAL PROJECT RESULTS ---')
print(f'Test Accuracy: {final_acc:.2f}%')
print(f'Total Images Tested: {total}')