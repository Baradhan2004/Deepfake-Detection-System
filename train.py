import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def main():

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------- CONFIG ----------------
    batch_size = 32
    image_size = 224
    epochs = 15

    # ---------------- TRANSFORMS ----------------
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ---------------- DATASET (FIXED) ----------------
    train_dataset = datasets.ImageFolder("train", transform=train_transform)
    val_dataset = datasets.ImageFolder("test", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ---------------- MODEL ----------------
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze backbone initially
    for param in model.features.parameters():
        param.requires_grad = False

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)

    model = model.to(device)

    # ---------------- LOSS & OPTIMIZER ----------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=2
    )

    best_val_acc = 0
    patience = 5
    counter = 0

    # ---------------- TRAINING ----------------
    for epoch in range(epochs):

        # 🔥 EARLY UNFREEZE (KEY)
        if epoch == 3:
            print("\n🔓 Unfreezing backbone for fine-tuning...")

            for param in model.features.parameters():
                param.requires_grad = True

            optimizer = optim.Adam(model.parameters(), lr=1e-5)

        # ---- TRAIN ----
        model.train()
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # ---- VALIDATION ----
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        # ---- SAVE BEST ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("🔥 Best model saved")
            counter = 0
        else:
            counter += 1

        # ---- EARLY STOP ----
        if counter >= patience:
            print("⛔ Early stopping triggered")
            break

    print("\n✅ Training complete")
    print("Best Validation Accuracy:", best_val_acc)


if __name__ == "__main__":
    main()
