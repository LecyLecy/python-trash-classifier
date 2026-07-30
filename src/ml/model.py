import torch
import torch.nn as nn
from torchvision import models

def create_model(
    model_name: str,
    num_classes: int,
    *,
    pretrained: bool = False,
) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None

    if model_name == "resnet18":
        m = models.resnet18(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m

    raise ValueError(f"Unsupported model: {model_name}")

@torch.no_grad()
def predict_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(x)
