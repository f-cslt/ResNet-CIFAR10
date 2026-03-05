import torch.nn as nn
from torchvision import models


def build_model(num_classes=10):

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
   
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model
