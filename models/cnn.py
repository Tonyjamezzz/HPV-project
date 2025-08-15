import torch
from torch import nn
from torchvision import models


class ResNetClassifier(nn.Module):
    """ResNet18-based classifier for normal vs abnormal classification.

    The model loads a ResNet18 backbone, replaces the final fully connected layer
    to match the number of classes (default: 2) and returns class probabilities.
    Images are expected to be 3-channel tensors of shape (N, 3, 224, 224).
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class probabilities.

        Args:
            x: Input tensor of shape (N, 3, 224, 224)

        Returns:
            Tensor of shape (N, num_classes) with probabilities for each class.
        """
        logits = self.backbone(x)
        probs = torch.softmax(logits, dim=1)
        return probs


__all__ = ["ResNetClassifier"]
