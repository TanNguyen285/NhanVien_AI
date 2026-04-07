import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TAFE(nn.Module):
    def __init__(self, in_channels):
        super(TAFE, self).__init__()
        # α, β là learnable scalar weights (không constrain)
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.beta = nn.Parameter(torch.tensor([0.5]))
        
        # Nhánh 1
        self.conv_branch1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gelu = nn.GELU()
        self.conv1x1_refine = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Nhánh 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

    def forward(self, x):
        o1 = self.conv_branch1(x)
        o2 = self.branch2(x)
        
        # [FIX 1]: Dùng đúng Variance (unbiased=False để chia cho HW)
        Os = torch.mean(o1, dim=(2, 3), keepdim=True)
        Ov = torch.var(o1, dim=(2, 3), keepdim=True, unbiased=False) 
        
        # Eq 4 & 5
        O = self.alpha * Os + self.beta * Ov
        v1 = self.conv1x1_refine(self.gelu(O)) * o1  # [FIX 2]: Giữ nguyên, không Sigmoid
        
        return torch.cat([v1, o2], dim=1)

class DCIF(nn.Module):
    def __init__(self, in_channels):
        super(DCIF, self).__init__()
        # [FIX 3]: Conv 7x7 chạy trên HxW, in_channels = 2 (mean + max)
        self.conv_attn = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
        # Channel-wise MLP cho kappa
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // 2, in_channels)
        
    def forward(self, x):
        # [FIX 3]: Spatial attention đúng chuẩn CBAM logic
        # Gộp theo chiều Channel (dim=1) để giữ lại không gian HxW
        r1 = torch.mean(x, dim=1, keepdim=True)        # (B, 1, H, W)
        r2 = torch.max(x, dim=1, keepdim=True)[0]      # (B, 1, H, W)
        R = torch.cat([r1, r2], dim=1)                 # (B, 2, H, W)
        
        # Tính Spatial Attention map
        eta = torch.sigmoid(self.conv_attn(R))         # (B, 1, H, W)
        theta = eta * x                                # (B, C, H, W)
        
        # Eq 12 & 13: Tính K = FC(ReLU(FC(kappa)))
        # Giả định kappa được lấy từ GAP của theta theo chiều không gian
        kappa = F.adaptive_avg_pool2d(theta, 1).flatten(1) # (B, C)
        K = self.fc2(self.relu(self.fc1(kappa)))           # (B, C)
        
        return K, theta # Trả về K (vector) và theta (feature map) tùy logic của lớp classifier

class TKFNet(nn.Module):
    def __init__(self, num_classes=7):
        super(TKFNet, self).__init__()
        backbone = models.resnet18(pretrained=True)
        # Giữ lại các layer trước GAP và FC
        self.features = nn.Sequential(*list(backbone.children())[:-2]) # Output: 512 channels
        
        self.tafe = TAFE(512)
        self.dcif = DCIF(1024) # TAFE concat v1 (512) và o2 (512) = 1024
        
        # [FIX 4]: Tuân thủ đúng Eq 14: Logits = FC(Flatten(GAP(K)))
        # Tùy thuộc vào việc K ở paper là feature map hay vector. 
        # Nếu K là vector (do qua FC của kappa), thì không cần GAP nữa.
        # Ở đây tôi thiết kế bộ Classifier cuối cùng nhận vector từ DCIF.
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.tafe(x)
        K, _ = self.dcif(x)
        
        # Eq 14: Tính Logits
        logits = self.classifier(K)
        return logits