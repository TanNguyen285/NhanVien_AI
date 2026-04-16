import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Union

class DepthwiseSeparableConv(nn.Module):
    """
    Tối ưu hóa tài nguyên bằng cách tách biệt tích chập không gian (spatial) 
    và tích chập kênh (channel). Thường dùng trong các kiến trúc di động.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) # Thêm BN để training ổn định hơn
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class LightAttNet(nn.Module):
    """
    LightAttNet: Kiến trúc mạng nhẹ kết hợp Attention Mechanism.
    Dựa trên MobileNetV2 backbone để trích xuất đặc trưng và 
    Multi-head Attention để tinh chỉnh các đặc trưng quan trọng.
    """
    def __init__(self, num_classes: int, dropout_rate: float = 0.3):
        super(LightAttNet, self).__init__()
        
        # 1. Feature Extractor (Backbone)
        # Sử dụng MobileNetV2 lên đến layer index 14 (output: 96 channels, 14x14)
        v2_features = models.mobilenet_v2(weights='DEFAULT').features
        self.backbone = nn.Sequential(*list(v2_features.children())[:14])
        
        # 2. Patch-based Feature Refinement
        self.patch_refinement = nn.Sequential(
            nn.ZeroPad2d(1),
            DepthwiseSeparableConv(96, 256, kernel_size=3, stride=2),
            DepthwiseSeparableConv(256, 256, kernel_size=3, stride=2),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        
        # 3. Aggregation & Feature Compression
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.feature_projector = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32)
        )
        
        # 4. Attention Mechanism
        self.attention_block = nn.MultiheadAttention(
            embed_dim=32, 
            num_heads=1, 
            batch_first=True
        )
        
        # 5. Output Head
        self.classifier = nn.Linear(32, num_classes)
        
        # Tự động khởi tạo trọng số cho các layer mới
        self._initialize_weights()

    def _initialize_weights(self):
        """Khởi tạo trọng số chuẩn cho các layer không phải backbone."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trích xuất đặc trưng bậc thấp (Backbone)
        x = self.backbone(x)           # Input (B, 3, 224, 224) -> (B, 96, 14, 14)
        
        # Tinh chỉnh đặc trưng cục bộ
        x = self.patch_refinement(x)   # (B, 96, 14, 14) -> (B, 256, 2, 2)
        
        # Nén đặc trưng
        x = self.global_pool(x).view(x.size(0), -1) 
        x = self.dropout(x)
        x = self.feature_projector(x)  # (B, 256) -> (B, 32)
        
        # Self-Attention (Yêu cầu input dạng sequence: B, L, D)
        x_seq = x.unsqueeze(1) 
        attn_out, _ = self.attention_block(x_seq, x_seq, x_seq)
        x = attn_out.squeeze(1)
        
        # Phân loại
        return self.classifier(x)