import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Lists for datasets and models
DATASETS = []
MODELS = []

# Hyperparmeters
EPOCHS = 100

# Loading models
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
resnet50 = models.resnet50(pretrained=True)

MODELS.append(alexnet)
MODELS.append(vgg16)
MODELS.append(resnet50)

print(alexnet)
print(vgg16)

for model in MODELS:
    for epoch in range(EPOCHS):
        pass
