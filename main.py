import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms, datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Lists for datasets and models
DATASETS = []
MODELS = [
    #models.alexnet(pretrained=True),
    models.vgg16(pretrained=True),
    #models.resnet50(pretrained=True)
]

# Hyperparmeters
EPOCHS = 3
BATCH_SIZE = 100


# CIFAR-10 Data Loaders
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

num_classes = len(train_set.classes)
for model in MODELS:
    if isinstance(model, models.AlexNet):
        dropout = 0.5
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        print("Classifier for model has been changed.")

        # Freezing the feature extraction layers
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif isinstance(model, models.VGG):
        dropout = 0.5
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        print("Classifier for model has been changed.")

        # Freezing the feature extraction layers
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif isinstance(model, models.ResNet):
        dropout = 0.5
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        print("Classifier for model has been changed.")

        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True


    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    # Training and Validation
    print("Training starts")
    for epoch in range(EPOCHS):
        print("Epoch", epoch)
        model.train()
        running_loss = 0.0
        for batch, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            print("Batch:", batch)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader)}, '
              f'Validation Loss: {val_loss/len(test_loader)}, '
              f'Accuracy: {100 * correct / total}%')
