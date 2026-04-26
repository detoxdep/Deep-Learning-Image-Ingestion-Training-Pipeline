import torch
import torch.nn as nn
import torch.optim as optim
from Models import ModelA, ModelB
from DataLoader import train_loader, val_loader, test_loader
from sklearn.metrics import classification_report, confusion_matrix

def run_full_experiment(model_class, name, device, epochs=10):
    print(f"\n" + "="*50)
    print(f" EXPERIMENT: {name}")
    print("="*50)

    # 1. Initialize Model, Loss, and Optimizer
    model = model_class.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 2. Training Loop
    print(f"Training {name} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation Check
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"[{name}] Epoch {epoch+1}/{epochs} | Val Acc: {100*correct/total:.2f}%")

    # 3. Save the Weights (The "Brain")
    save_filename = f"{name}_weights.pth"
    torch.save(model.state_dict(), save_filename)
    print(f"\nDONE: Weights saved successfully as {save_filename}")

    # 4. Final Evaluation (Precision, Recall, F1)
    print(f"\nGenerating Detailed Research Report for {name}...")
    all_preds, all_labels = [], []
    classes = ['glass', 'metal', 'paper', 'plastic']
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Output Classification Report
    print("\n--- CLASSIFICATION METRICS ---")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Output Confusion Matrix (Identifies the Faults)
    print("--- CONFUSION MATRIX ---")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"{'':<10} " + " ".join([f"{c:<8}" for c in classes]))
    for i, row in enumerate(cm):
        print(f"{classes[i]:<10} {row}")

def main():
    # Setup Device (Mac GPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"System Initialized. Using Device: {device}")

    # Run everything for Model A
    run_full_experiment(ModelA(), "Model_A", device)

    # Run everything for Model B
    run_full_experiment(ModelB(), "Model_B", device)

if __name__ == "__main__":
    main()