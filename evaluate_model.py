import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from Models import ModelB
from DataLoader import test_loader

# 1. Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ModelB().to(device)
model.load_state_dict(torch.load('Model_B_weights.pth', map_location=device)) 
model.eval()

all_preds = []
all_labels = []

print("Gathering model predictions for detailed analysis...")

# 2. Collect all predictions vs actual labels
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # Move to CPU and convert to numpy for sklearn
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 3. Define your class names
classes = ['glass', 'metal', 'paper', 'plastic']

# 4. Generate the Research Data
print("\n" + "="*30)
print("   DETAILED RESEARCH REPORT")
print("="*30)

# This calculates Precision, Recall, and F1-Score for every class automatically
print(classification_report(all_labels, all_preds, target_names=classes))

print("\n--- CONFUSION MATRIX ---")
# This shows exactly which classes are getting mixed up
cm = confusion_matrix(all_labels, all_preds)
print(f"{'':<10} " + " ".join([f"{c:<8}" for c in classes]))
for i, row in enumerate(cm):
    print(f"{classes[i]:<10} {row}")