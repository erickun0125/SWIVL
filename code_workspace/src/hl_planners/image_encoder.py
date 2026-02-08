"""
Shared Image Encoder for High-Level Planners

ResNet18-based image encoder used by ACT, Diffusion Policy, and Flow Matching policies.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """
    ResNet18-based image encoder with ImageNet pretrained weights.
    Input: (batch, seq, 3, 96, 96)
    Output: (batch, seq, hidden_dim)
    """
    def __init__(self, hidden_dim: int, freeze_backbone: bool = False):
        super().__init__()
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove the final fully connected layer
        # ResNet18 outputs 512-dim features before fc layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove avgpool and fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Optionally freeze backbone weights
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Project to hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU()
        )

        # ImageNet normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, 3, H, W), assumed to be in [0, 1] range
        batch, seq, c, h, w = x.shape
        x = x.view(batch * seq, c, h, w)

        # Apply ImageNet normalization
        x = (x - self.mean) / self.std

        # Extract features
        feat = self.backbone(x)  # (batch*seq, 512, h', w')
        feat = self.avgpool(feat)  # (batch*seq, 512, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (batch*seq, 512)
        feat = self.fc(feat)  # (batch*seq, hidden_dim)

        return feat.view(batch, seq, -1)
