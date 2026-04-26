import torch
from Models import ModelB
from DataLoader import test_loader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ModelB().to(device)
model.load_state_dict(torch.load('Model_B_weights.pth', map_location=device)) 
model.eval()

# Tracking variables
classes = ['glass', 'metal', 'paper', 'plastic']
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        
        # Collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# Print the breakdown
print(f"{'Class':<10} | {'Accuracy':<10}")
print("-" * 25)
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"{classname:<10} | {accuracy:.2f}%")