import torch
import torch.nn as nn
from torchvision import models

# ==========================================
# Separable Convolution
# ==========================================
class SeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution để giảm tham số và tính toán.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        # Depthwise: mỗi channel học một filter riêng
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )

        # Pointwise: kết hợp thông tin giữa các channels
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(x)
        return x


# ==========================================
# PAttLite Model (LightAttNet)
# ==========================================
class PAttLite(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.2):
        super().__init__()

        # ===== Backbone (MobileNetV2) =====
        # Lấy đến layer thứ 14 của MobileNetV2 features
        v2_model = models.mobilenet_v2(weights="DEFAULT").features
        self.backbone = nn.Sequential(*list(v2_model.children())[:14])
        # Output mong đợi: (Batch, 96, H', W') -> Với input 112x112 là (B, 96, 7, 7)

        # ===== Patch Extraction =====
        self.patch_extract = nn.Sequential(
            nn.ZeroPad2d(1),  # (7x7) → (9x9)
            SeparableConv2d(96, 256, kernel_size=3, stride=2),  # → (4x4)
            SeparableConv2d(256, 256, kernel_size=2, stride=2), # → (2x2)
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # ===== Head =====
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)

        self.pre_class = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32)
        )

        # Attention cơ chế Multihead (ở đây dùng 1 token duy nhất)
        self.attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=1,
            batch_first=True
        )

        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        # 1. Trích xuất đặc trưng từ Backbone
        x = self.backbone(x)

        # 2. Xử lý Patch/Giảm chiều không gian
        x = self.patch_extract(x)

        # 3. Global Average Pooling & Flatten
        x = self.gap(x).flatten(1) 

        x = self.dropout(x)

        # 4. Dense Layer (Giảm dim xuống 32)
        x = self.pre_class(x)

        # 5. Attention Mechanism
        # Thêm chiều sequence: (Batch, 32) -> (Batch, 1, 32)
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = attn_out.squeeze(1)

        # 6. Phân lớp
        return self.classifier(x)

# ==========================================
# Kiểm tra nhanh model (Sanity Check)
# ==========================================
if __name__ == "__main__":
    # Khởi tạo model
    model = PAttLite(num_classes=7)
    model.eval()

    # Tạo input giả lập (Batch size=1, Channels=3, H=112, W=112)
    dummy_input = torch.randn(1, 3, 112, 112)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Mong đợi: torch.Size([1, 7])
    
    # Tính tổng số tham số
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")