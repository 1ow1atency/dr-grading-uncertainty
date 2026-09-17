"""EfficientNet-B0 backbone DR severity classifier with MC Dropout uncertainty estimation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

NUM_CLASSES = 5


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 with its classifier head replaced by Dropout(0.4) + Linear(features, num_classes)."""
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """Eval mode (frozen BatchNorm running stats) with Dropout layers forced back to
    train mode, so MC Dropout samples stay stochastic at inference time."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    return model


@torch.no_grad()
def predict_with_uncertainty(
    model: nn.Module,
    image: torch.Tensor,
    n_samples: int = 30,
) -> tuple[torch.Tensor, float]:
    """Draws n_samples MC Dropout forward passes for a single (already transformed)
    image and returns (mean class probabilities, predictive entropy).

    Predictive entropy -sum(p * log(p)) over the mean probabilities is the
    uncertainty score: it's high both when the samples disagree with each other
    and when they agree on a flat distribution, capturing overall predictive
    uncertainty rather than just sample-to-sample variance.
    """
    enable_mc_dropout(model)
    device = next(model.parameters()).device

    if image.dim() == 3:
        image = image.unsqueeze(0)
    batch = image.to(device).repeat(n_samples, 1, 1, 1)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)

    mean_probs = probs.mean(dim=0)
    entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum()

    return mean_probs.cpu(), entropy.item()
