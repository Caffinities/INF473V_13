
import torchvision
import torch.nn as nn


class ResNext(nn.Module):
    def __init__(self, num_classes, frozen=False):
        super().__init__()
        self.backbone = torchvision.models.resnext50_32x4d(weights='DEFAULT')
        self.backbone.fc = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(nn.Linear(2048, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, num_classes),)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x