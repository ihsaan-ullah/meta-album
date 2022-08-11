import torch
import torch.nn as nn
from torchvision import models
from avalanche.models.dynamic_modules import MultiTaskModule, \
    MultiHeadClassifier


class Identity(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x


class MultiHeadResnet18(MultiTaskModule):
    
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "MultiHeadResnet18"
        self.feature_extractor = models.resnet18(pretrained=True)

        for name, param in self.feature_extractor.named_parameters():
            if name.startswith(("fc", "layer4.1.bn2", "layer4.1.conv2")):
                param.requires_grad = True
            else:
                param.requires_grad = False

        fc_in_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()

        self.classifier = MultiHeadClassifier(fc_in_features)

    def forward(self, 
                x: torch.tensor, 
                task_labels) -> torch.tensor:
        x = self.feature_extractor(x)
        x = x.squeeze()
        x = self.classifier(x, task_labels)
        return x
