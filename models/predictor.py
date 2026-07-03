import torch
import torch.nn as nn

class DegradationPredictor(nn.Module):
    def __init__(self, in_channels=1, num_classes=2): 
        super(DegradationPredictor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # /4
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # (B, 64, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits
