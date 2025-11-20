import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import FSWDataset
from models.architecture import CNN_BiLSTM
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
NUM_CLASSES = 7
DATA_PATH = './data'  # Path to your dataset folder

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Load Dataset
    full_dataset = FSWDataset(root_dir=DATA_PATH, num_frames=15, use_segmentation=True)
    
    # Train/Val Split (Example: 80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Initialize Model
    model = CNN_BiLSTM(num_classes=NUM_CLASSES).to(device)
    
    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # 4. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        for frames, labels in tqdm(train_loader, desc="Training"):
            frames, labels = frames.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        
        # 5. Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                outputs = model(frames)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        scheduler.step(val_acc)

    # Save Model
    torch.save(model.state_dict(), "fsw_action_model.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()
