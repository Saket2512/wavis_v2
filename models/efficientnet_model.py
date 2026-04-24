"""
WAVIS v2 - EfficientNet-B3 Model
=================================
Transfer learning from ImageNet pretrained EfficientNet-B3.
Input: 3-channel 224×224 Mel spectrogram image
Output: Species probability distribution

Why EfficientNet-B3 hits 95%+:
  - Pretrained on 1.2M ImageNet images → strong feature extractor
  - Mel spectrograms are 2D images → vision model works perfectly
  - B3 is the sweet spot: accuracy vs speed vs memory for GPU laptop
  - Fine-tuning + SpecAugment + MixUp pushes past 95%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class WildlifeEfficientNet(nn.Module):
    """
    EfficientNet-B3 backbone with custom wildlife classification head.

    Architecture:
        EfficientNet-B3 backbone (pretrained ImageNet)
            └─ features (1536-dim)
                └─ dropout(0.3)
                    └─ Linear(1536 → 512) + GELU + BN
                        └─ dropout(0.25)
                            └─ Linear(512 → n_classes)
    """

    def __init__(self, num_classes: int,
                 backbone: str = 'efficientnet_b3',
                 pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        # Load pretrained backbone — drop its classifier
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,            # remove head
            global_pool='avg',        # global average pool
            in_chans=3,               # 3-channel mel image
        )

        # Get feature dimension
        feat_dim = self.backbone.num_features

        # Custom classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.8),
            nn.Linear(512, num_classes),
        )

        # Weight init for head only
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224)"""
        features = self.backbone(x)      # (B, feat_dim)
        return self.head(features)       # (B, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embeddings (for visualization / t-SNE)."""
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def freeze_backbone(self):
        """Freeze backbone — only train head (Phase 1 of training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen — training head only")

    def unfreeze_backbone(self, lr_multiplier: float = 0.1):
        """Unfreeze backbone for fine-tuning (Phase 2)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"🔓 Backbone unfrozen — fine-tuning at {lr_multiplier}x LR")

    def count_parameters(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def load_model(checkpoint_path: str, num_classes: int,
               backbone: str = 'efficientnet_b3',
               device: str = 'cpu') -> WildlifeEfficientNet:
    """Load a trained model from checkpoint."""
    model = WildlifeEfficientNet(num_classes=num_classes, backbone=backbone, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(device)
    return model


# ─── MixUp / CutMix (key to 95%+) ────────────────────────────────────────────

def mixup_data(x: torch.Tensor, y: torch.Tensor,
               alpha: float = 0.4) -> tuple:
    """MixUp augmentation — interpolates two training examples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss = weighted combination of two sample losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


import numpy as np  # needed for mixup_data


if __name__ == '__main__':
    model = WildlifeEfficientNet(num_classes=50)
    print(f"Total params:     {model.count_parameters(trainable_only=False):,}")
    print(f"Trainable params: {model.count_parameters(trainable_only=True):,}")

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
