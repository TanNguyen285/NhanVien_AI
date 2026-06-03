import torch.nn as nn
from torchvision import models


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.pointwise(self.depthwise(x)))


class PAttLite(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.2):
        super().__init__()

        # Backbone MobileNetV2 (đến layer 14)
        v2_features = models.mobilenet_v2(weights="DEFAULT").features
        self.backbone = nn.Sequential(*list(v2_features.children())[:14])
        # Output: (B, 96, 7, 7) với input 112×112

        # Patch Extraction
        self.patch_extract = nn.Sequential(
            nn.ZeroPad2d(1),                                          # 7→9
            SeparableConv2d(96, 256, kernel_size=3, stride=2),        # 9→4
            SeparableConv2d(256, 256, kernel_size=2, stride=2),       # 4→2
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.dropout   = nn.Dropout(dropout_rate)
        self.pre_class = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32)
        )
        self.attention  = nn.MultiheadAttention(embed_dim=32, num_heads=1, batch_first=True)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.patch_extract(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        x = self.pre_class(x)
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = attn_out.squeeze(1)
        return self.classifier(x)