import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super().__init__()
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base_model.children())[:-1])  # output: [B, 512, 1, 1]
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.cnn(x).view(B, T, -1)
        lstm_out, _ = self.lstm(feat)
        out = self.fc(lstm_out[:, -1])
        return out
