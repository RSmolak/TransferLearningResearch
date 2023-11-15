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
    models.alexnet(pretrained=True),
    models.vgg16(pretrained=True),
    models.resnet50(pretrained=True)
]

# Hyperparmeters
EPOCHS = 100
BATCH_SIZE = 64


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


for model in MODELS:
    # Replace the final layer for CIFAR-10
    if isinstance(model, models.AlexNet) or isinstance(model, models.VGG):
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 10)
    elif isinstance(model, models.ResNet):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training and Validation
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
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
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader)}, '
              f'Validation Loss: {val_loss/len(test_loader)}, '
              f'Accuracy: {100 * correct / total}%')
