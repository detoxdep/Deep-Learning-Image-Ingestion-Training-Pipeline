import torch
import torch.nn as nn
import torch.optim as optim
from Models import ModelA, ModelB
from DataLoader import train_loader, val_loader

# 1. Setup Device (Uses Mac GPU if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

models_to_train = [
    {'name': 'Model_A', 'class': ModelA()},
    {'name': 'Model_B', 'class': ModelB()}
]
# 2. Initialize Model, Loss, and Optimizer
for item in models_to_train:
    model = item['class'].to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training Loop
    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
        
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
        
            # Backward pass (The "Learning" part)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ----- Accuracy Check on Validation Set -----
        model.eval() 
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/10] complete. Val Accuracy: {accuracy:.2f}%')
    

    # --- ADD THE SAVE COMMAND ---
    # If you want to start training again, comment out this section, and run vs code from the beguining
    # --- FIXED SAVE COMMAND ---
    # This uses the 'name' from your list (Model_A or Model_B)
    save_filename = f"{item['name']}_weights.pth"
    torch.save(model.state_dict(), save_filename)
    print(f"Finished {item['name']}! Saved as {save_filename}")