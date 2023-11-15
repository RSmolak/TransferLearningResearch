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

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(EPOCHS):
        pass
