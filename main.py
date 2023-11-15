import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATASETS = []
MODELS = []

EPOCHS = 100

resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
MODELS.append(resnet50)


for model in MODELS:
    for epoch in range(EPOCHS):
        pass
