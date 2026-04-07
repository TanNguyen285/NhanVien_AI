import torch
import torch.nn as nn
from torchvision import models

# ==========================================
# KIẾN TRÚC MÔ HÌNH (CHUẨN MOBILENETV2)
# ==========================================

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.pointwise(self.depthwise(x)))

class PAttLite(nn.Module):
    def __init__(self, num_classes=8, dropout_rate=0.2):
        super(PAttLite, self).__init__()
        
        # Load MobileNetV2 Pre-trained
        v2_model = models.mobilenet_v2(weights='DEFAULT').features
        self.backbone = nn.Sequential(*list(v2_model.children())[:14])
        
        # Patch Extraction (Input 96 channels từ V2)
        self.patch_extract = nn.Sequential(
            nn.ZeroPad2d(1),
            SeparableConv2d(96, 256, kernel_size=4, stride=4, padding=0),
            SeparableConv2d(256, 256, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.ReLU()
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.pre_class = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=1, batch_first=True)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.backbone(x)           # (Batch, 96, 14, 14)
        x = self.patch_extract(x)      # (Batch, 256, 2, 2)
        x = self.gap(x).view(x.size(0), -1) 
        x = self.dropout(x)
        x = self.pre_class(x)
        
        x_in = x.unsqueeze(1) 
        attn_out, _ = self.attention(x_in, x_in, x_in)
        x = attn_out.squeeze(1)
        
        return self.classifier(x)