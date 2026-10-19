import torch
import torch.nn as nn
import numpy as np
class CNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, image):
        output = self.pool1(self.relu1(self.bn1(self.conv1(image))))
        output = self.pool2(self.relu2(self.bn2(self.conv2(output))))
        output = self.pool3(self.relu3(self.bn3(self.conv3(output))))

        output = self.average_pool(output)
        output = self.flatten(output)
        output = self.dropout(output)
        output = self.fc(output)

        return output

