import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, classification_report
import os

def evaluate():
    print("Loading test dataset...")
    data_dir = 'test'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Classes detected: {test_dataset.classes}")
    
    print("Loading model...")
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    model.load_state_dict(torch.load("best_multigan_fast.pth", map_location=device))
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    print("Running evaluation (this may take a moment depending on dataset size)...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            # Assume class index 1 is 'fake' (or the positive class)
            all_probs.extend(probs[:, 1].cpu().numpy())

    # Calculate metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS:")
    print("="*50)
    
    f1 = f1_score(all_labels, all_preds)
    print(f"F1 Score: {f1:.4f}")
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
    
    # Save plots
    print("\nGenerating plots...")
    os.makedirs('evaluation_results', exist_ok=True)
    
    # 1. ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('evaluation_results/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig('evaluation_results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation complete! Check the 'evaluation_results' folder for the plots.")

if __name__ == '__main__':
    evaluate()
