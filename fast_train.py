import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import time

# ==========================================
# 1. Configuration (Optimized for RTX & Heat Reduction)
# ==========================================
DATA_DIR = r'C:\Users\BARADHAN\.gemini\antigravity\scratch\deepfake-detector' 
BATCH_SIZE = 64 # High batch size is good for RTX, reduces training time
EPOCHS = 5 # Fewer epochs needed with EfficientNet
LEARNING_RATE = 0.001

# Enable GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# ==========================================
# 2. Data Loading (Using your train/test folders)
# ==========================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([ # Using 'test' instead of 'val' based on your folder
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train', 'test']}

# Increased num_workers helps feed the GPU faster, keeping training time short
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
               for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
print(f"Classes found: {class_names}")

# ==========================================
# 3. Model: EfficientNet-B0 (Fastest, Coolest, High Accuracy)
# ==========================================
# EfficientNet-B0 gives extremely high accuracy but uses far fewer parameters than ResNet
# This prevents your laptop from overheating!
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE) # AdamW converges faster

# Use mixed precision (AMP) - this halves the memory and doubles speed on RTX!
scaler = GradScaler()

# ==========================================
# 4. Training Loop
# ==========================================
def train_model(model, criterion, optimizer, num_epochs=5):
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Autocast runs the forward pass in FP16 (Mixed Precision) - Massive speedup
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # Save as PyTorch format (.pth)
                torch.save(model.state_dict(), 'best_multigan_fast.pth')

    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Test Accuracy: {best_acc:4f}')
    return model

if __name__ == '__main__':
    print("Starting ultra-fast RTX training...")
    trained_model = train_model(model, criterion, optimizer, num_epochs=EPOCHS)
    print("Saved as 'best_multigan_fast.pth'")
