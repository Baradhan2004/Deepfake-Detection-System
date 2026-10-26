import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
# Change this to the path where your dataset is stored.
# The dataset should have subfolders: 'train/real', 'train/fake', 'val/real', 'val/fake'
DATA_DIR = 'path_to_your_dataset' 
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Check if your NVIDIA RTX GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# ==========================================
# 2. Data Augmentation and Loading
# ==========================================
# Resize images, convert to tensors, and normalize them based on ImageNet standards
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(f"Classes found: {class_names}")

# ==========================================
# 3. Model Setup (Transfer Learning)
# ==========================================
# We use a pre-trained ResNet50 model, which is highly accurate for image classification
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freeze early layers if you want to speed up training, but for fine-tuning we leave them unfrozen
# Replace the final fully connected layer to output 2 classes (Real vs Fake)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 4. Training Loop
# ==========================================
def train_model(model, criterion, optimizer, num_epochs=10):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_multigan_model.pth')

        print()

    print(f'Best Validation Accuracy: {best_acc:4f}')
    return model

if __name__ == '__main__':
    # Start training
    print("Starting training process...")
    trained_model = train_model(model, criterion, optimizer, num_epochs=EPOCHS)
    print("Training complete. Best model saved as 'best_multigan_model.pth'")
