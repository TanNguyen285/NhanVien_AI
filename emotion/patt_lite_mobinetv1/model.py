import torch
import torch.nn as nn

def conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv_dw(in_channels, out_channels, stride):
    return nn.Sequential(
        # Depthwise
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        # Pointwise
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # Depthwise: Conv -> BN -> ReLU
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise: Conv -> BN -> ReLU
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x

class PAttLite_V1_Final(nn.Module):
    def __init__(self, num_classes=8, dropout_rate=0.2):
        super(PAttLite_V1_Final, self).__init__()
        
        # --- 1. BACKBONE (MobileNetV1 trích xuất đến layer -29) ---
        # Feature map đầu ra dự kiến: [Batch, 512, 14, 14]
        self.backbone = nn.Sequential(
            conv_bn(3, 32, 2),        # 112x112
            conv_dw(32, 64, 1),       # 112x112
            conv_dw(64, 128, 2),      # 56x56
            conv_dw(128, 128, 1),     # 56x56
            conv_dw(128, 256, 2),     # 28x28
            conv_dw(256, 256, 1),     # 28x28
           # conv_dw(256, 512, 2),     # 14x14
        )

    # --- TỰ ĐỘNG LẤY SỐ CHANNELS ĐẦU RA CỦA BACKBONE ---
        # Lấy layer cuối cùng của backbone, xem nó xuất ra bao nhiêu channels
        # Tìm lớp BatchNorm hoặc Conv cuối cùng để lấy 'num_features' hoặc 'out_channels'
        backbone_out_channels = None
        for module in reversed(self.backbone):
            if hasattr(module, 'out_channels'): # Nếu là lớp Conv
                backbone_out_channels = module.out_channels
                break
            elif isinstance(module, nn.Sequential): # Nếu là khối conv_dw (thường là Sequential)
                # Thử tìm sâu bên trong khối Sequential đó
                for sub_m in reversed(module):
                    if hasattr(sub_m, 'out_channels'):
                        backbone_out_channels = sub_m.out_channels
                        break
                if backbone_out_channels: break

        # Nếu không tìm thấy bằng cách duyệt, ta dùng phương pháp "Dummy Input" (Chắc chắn nhất)
        if backbone_out_channels is None:
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 112, 112)
                dummy_out = self.backbone(dummy_input)
                backbone_out_channels = dummy_out.size(1)

        print(f"🔍 Tự động phát hiện Backbone đầu ra: {backbone_out_channels} channels")

        # --- 2. PATCH EXTRACTION ---
        # Bây giờ ta dùng biến 'backbone_out_channels' thay vì ghi số chết
        self.patch_extraction = nn.Sequential(
            SeparableConv2d(backbone_out_channels, 256, kernel_size=4, stride=4, padding=1), 
            SeparableConv2d(256, 256, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        
        # --- 3. PRE-CLASSIFICATION ---
        self.pre_classification = nn.Sequential(
            nn.Linear(256, 32, bias=False), # Dùng bias=False vì có BN ngay sau
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        
        # --- 4. ATTENTION ---
        # TF Attention mặc định là Luong (dot product), MultiheadAttention với 1 head là tương đương
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=1, batch_first=True)
        
        # --- 5. CLASSIFICATION HEAD ---
        self.prediction_layer = nn.Linear(32, num_classes)

    def forward(self, x):
        # 1. Trích xuất đặc trưng
        x = self.backbone(x)
        x = self.patch_extraction(x)
        
        # 2. Global Average Pooling
        x = self.gap(x).view(x.size(0), -1) 
        x = self.dropout(x)
        
        # 3. Pre-classification
        x = self.pre_classification(x)
        
        # 4. Self-attention 
        # (Batch, 32) -> (Batch, 1, 32)
        x_in = x.unsqueeze(1) 
        # MultiheadAttention trả về (output, weights)
        attn_out, _ = self.attention(x_in, x_in, x_in)
        x = attn_out.squeeze(1)
        
        # 5. Output (Logits)
        return self.prediction_layer(x)