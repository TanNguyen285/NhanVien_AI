import torch
import torch.nn as nn
import torch.nn.functional as F

class ESHA(nn.Module):
    def __init__(self, dim, qk_dim=16, r=0.25, sr_ratio=2):
        super().__init__()
        self.rc = int(dim * r)
        self.qk_dim = qk_dim
        self.sr_ratio = sr_ratio
        
        # Input Projection: 3x3 Group Conv với 32 nhóm [cite: 312, 313]
        self.input_proj = nn.Conv2d(dim, qk_dim*2 + self.rc + (dim - self.rc), 
                                    kernel_size=3, padding=1, groups=32)
        
        # Spatial Reduction cho K và V để tăng tốc [cite: 320, 321]
        if sr_ratio > 1:
            self.sr_k = nn.Conv2d(qk_dim, qk_dim, kernel_size=sr_ratio, stride=sr_ratio, groups=qk_dim)
            self.sr_v = nn.Conv2d(self.rc, self.rc, kernel_size=sr_ratio, stride=sr_ratio, groups=self.rc)
        
        self.output_proj = nn.Conv2d(dim, dim, kernel_size=1) # [cite: 326]

    def forward(self, x):
        B, C, H, W = x.shape
        qkvu = self.input_proj(x)
        q, k, v, u = torch.split(qkvu, [self.qk_dim, self.qk_dim, self.rc, C - self.rc], dim=1)

        # Nhánh Attention xử lý 1/4 số kênh [cite: 246, 316]
        q_f = q.flatten(2).transpose(1, 2) 
        if self.sr_ratio > 1:
            k_f = self.sr_k(k).flatten(2).transpose(1, 2)
            v_f = self.sr_v(v).flatten(2).transpose(1, 2)
        else:
            k_f = k.flatten(2).transpose(1, 2)
            v_f = v.flatten(2).transpose(1, 2)

        attn = (q_f @ k_f.transpose(-2, -1)) * (self.qk_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        v_attn = (attn @ v_f).transpose(1, 2).reshape(B, self.rc, H, W)

        # Nhánh Local (U) giữ nguyên đặc trưng [cite: 323]
        out = torch.cat([v_attn, F.gelu(u)], dim=1)
        return self.output_proj(out)

class MicroViTBlock(nn.Module):
    def __init__(self, dim, use_esha=False, sr_ratio=2):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim) # Sử dụng BN thay vì LN để nhanh hơn [cite: 352]
        self.mixer = ESHA(dim, sr_ratio=sr_ratio) if use_esha else nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1), # Expansion ratio = 2 [cite: 349]
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class MicroViT(nn.Module):
    def __init__(self, variant='S2', num_classes=1000):
        super().__init__()
        # Cấu hình dựa trên Table I [cite: 329]
        cfgs = {
            'S1': {'c': [128, 256, 320], 'b': [2, 5, 5], 'sr': 2},
            'S2': {'c': [128, 256, 448], 'b': [2, 7, 5], 'sr': 2},
            'S3': {'c': [192, 384, 512], 'b': [3, 6, 6], 'sr': 1}
        }
        cfg = cfgs[variant]
        
        # Stem: giảm resolution 16x [cite: 331]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, cfg['c'][0], 3, stride=2, padding=1)
        )

        self.stage1 = nn.Sequential(*[MicroViTBlock(cfg['c'][0]) for _ in range(cfg['b'][0])])
        self.patch2 = nn.Conv2d(cfg['c'][0], cfg['c'][1], 3, stride=2, padding=1)
        self.stage2 = nn.Sequential(*[MicroViTBlock(cfg['c'][1]) for _ in range(cfg['b'][1])])
        self.patch3 = nn.Conv2d(cfg['c'][1], cfg['c'][2], 3, stride=2, padding=1)
        self.stage3 = nn.Sequential(*[MicroViTBlock(cfg['c'][2], use_esha=True, sr_ratio=cfg['sr']) for _ in range(cfg['b'][2])])
        
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cfg['c'][2], num_classes))

    def forward(self, x):
        return self.head(self.stage3(self.patch3(self.stage2(self.patch2(self.stage1(self.stem(x)))))))