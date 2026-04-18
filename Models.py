import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        
       #convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After two pools, 150x150 → 75x75 → 37x37
        self.fc1 = nn.Linear(32 * 37 * 37, 128)
        self.fc2 = nn.Linear(128, 4)   # 4 classes: paper, plastic, glass, metal

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After three pools: 150x150 → 75x75 → 37x37 → 18x18
        self.fc1 = nn.Linear(128 * 18 * 18, 256)
        self.fc2 = nn.Linear(256, 4)   # 4 classes: paper, plastic, glass, metal

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    # Quick test to see if models load
if __name__ == "__main__":
    model_a = ModelA()
    model_b = ModelB()
    print("Successfully initialized Model A and Model B!")
    
    # Create a fake 'image' to test the flow (1 image, 3 channels, 150x150)
    fake_image = torch.randn(1, 3, 150, 150)
    output = model_a(fake_image)
    print(f"Model A Test Output Shape: {output.shape} (Should be [1, 4])")