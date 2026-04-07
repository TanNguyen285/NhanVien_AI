import torch
import torch.nn as nn

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class EfficientBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_ratio=4): # Tăng lên 4 cho FER
        super(EfficientBlock, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion_ratio

        self.conv = nn.Sequential(
            # 1. Expansion
            conv_bn_relu(in_channels, hidden_dim, kernel_size=1),
            # 2. Depthwise
            conv_bn_relu(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim),
            # 3. Linear Projection
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EmotionCNN(nn.Module): # Đổi tên cho chuyên nghiệp
    def __init__(self, num_classes=6): # FER-2013 có 7 nhãn
        super(EmotionCNN, self).__init__()
        
        # Layer 1: Cửa ngõ (Giữ nguyên hoặc dùng stride=1 nếu ảnh 48x48)
        # Nếu dùng ảnh 48x48 của FER, đừng dùng stride=2 ở đây kẻo mất thông tin sớm
        self.stem = conv_bn_relu(3, 32, kernel_size=3, stride=1, padding=1) 

        self.blocks = nn.Sequential(
            # Layer 2-3: Học các nét cơ bản (cạnh, góc)
            EfficientBlock(32, 32, stride=1, expansion_ratio=2),
            EfficientBlock(32, 64, stride=2, expansion_ratio=4), 
            
            # Layer 4-5: Học tổ hợp (mắt, mũi, miệng)
            EfficientBlock(64, 64, stride=1, expansion_ratio=4),
            EfficientBlock(64, 128, stride=2, expansion_ratio=6),
            
            # Layer 6-7: Học ngữ cảnh cảm xúc cao cấp
            EfficientBlock(128, 128, stride=1, expansion_ratio=6),
            EfficientBlock(128, 256, stride=2, expansion_ratio=6),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.4), # Giảm nhẹ dropout nếu model nhỏ
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x

# Kiểm tra
if __name__ == "__main__":
    # FER-2013 có 7 class
    model = EmotionCNN(num_classes=6)
    
    # FER gốc là 48x48, nếu bạn dùng DataLoader mình viết ở trên thì input là 48x48
    test_input = torch.randn(1, 3, 48, 48) 
    output = model(test_input)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model Cảm Xúc đã sẵn sàng!")
    print(f"[*] Tổng tham số: {total_params / 1e6:.3f} M") # Sẽ rơi vào khoảng 0.5M - 0.8M
    print(f"[*] Output shape: {output.shape}") # [1, 7]