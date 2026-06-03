import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from datetime import datetime

# ==========================================
# 0. CẤU HÌNH & LOGGING
# ==========================================
LOG_FILE = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def write_log(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# Sửa lỗi Path: Sử dụng r"" (raw string) hoặc đổi dấu gạch chéo
H5_PATH = r"C:\Users\ThisPC\Documents\GitHub\NhanVien_AI_Emo\emodata_112x112_gray.h5"
NUM_CLASSES = 7
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Angry','Contempt','Disgust', 'Happy','Neutral','Sad', 'Surprise' ]

# ==========================================
# 1. CHUẨN BỊ DỮ LIỆU
# ==========================================
class EmotionDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        img = self.X[idx]
        if self.transform: img = self.transform(img)
        return img, self.y[idx]

if not os.path.exists(H5_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy file tại: {H5_PATH}")

write_log("📊 Đang tải dữ liệu từ file h5...")
with h5py.File(H5_PATH, 'r') as hf: 
    X_train, y_train = hf['X_train'][:], hf['y_train'][:]
    X_valid, y_valid = hf['X_val'][:], hf['y_val'][:]
    X_test,  y_test  = hf['X_test'][:], hf['y_test'][:]

X_train, y_train = shuffle(X_train, y_train, random_state=42)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# Transforms (Giữ nguyên logic của bạn)
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3), # Thêm dòng này: Biến 1 kênh -> 3 kênh
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3), # Thêm dòng này luôn cho đồng bộ
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(EmotionDataset(X_train, y_train, train_transforms), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(EmotionDataset(X_valid, y_valid, val_test_transforms), batch_size=BATCH_SIZE)
test_loader  = DataLoader(EmotionDataset(X_test, y_test, val_test_transforms), batch_size=BATCH_SIZE)

import torch
import torch.nn as nn
from torchvision import models


# ==========================================
# Separable Convolution
# ==========================================
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )

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
        v2_model = models.mobilenet_v2(weights="DEFAULT").features
        self.backbone = nn.Sequential(*list(v2_model.children())[:14])
        # Output: (B, 96, 7, 7) với input 112x112

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

        self.attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=1,
            batch_first=True
        )

        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        # Backbone
        x = self.backbone(x)

        # Patch extraction
        x = self.patch_extract(x)

        # Global pooling
        x = self.gap(x).flatten(1)  # an toàn hơn view

        x = self.dropout(x)

        # Dense
        x = self.pre_class(x)

        # Attention (1 token)
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)

        x = attn_out.squeeze(1)

        # Classifier
        return self.classifier(x)

# ==========================================
# 3. HÀM TRỰC QUAN HÓA (CONFUSION MATRIX)
# ==========================================
def plot_confusion_matrix(model, loader, phase_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix - {phase_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"cm_{phase_name}.png")
    plt.close()

# ==========================================
# 4. HUẤN LUYỆN
# ==========================================
def train_model(model, optimizer, criterion, epochs, patience, phase_name):
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss, train_corr = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_corr += torch.sum(preds == labels.data)

        train_acc = train_corr.double() / len(train_loader.dataset)
        
        model.eval()
        val_corr = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corr += torch.sum(preds == labels.data)
        
        val_acc = val_corr.double() / len(val_loader.dataset)
        
        msg = f"[{phase_name}] Epoch {epoch+1:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        write_log(msg)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                write_log("=> Early Stopping!")
                break
    
    model.load_state_dict(best_model_wts)
    return model

# Khởi chạy
model = PAttLite().to(DEVICE)

write_log("\n=== PHASE 1: WARM-UP ===")
for param in model.backbone.parameters(): param.requires_grad = False
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
model = train_model(model, optimizer, nn.CrossEntropyLoss(weight=class_weights_tensor), 15, 4, "Phase 1")

write_log("\n=== PHASE 2: FINE-TUNING ===")
for param in model.parameters(): param.requires_grad = True
for m in model.backbone.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval(); [p.requires_grad_(False) for p in m.parameters()]

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
model = train_model(model, optimizer, nn.CrossEntropyLoss(weight=class_weights_tensor), 35, 10, "Phase 2")

# ==========================================
# 5. KẾT THÚC
# ==========================================
write_log("\n🎨 Đang tạo Confusion Matrix cho tập Test...")
plot_confusion_matrix(model, test_loader, "Final_Test")
torch.save(model.state_dict(), 'PAtt_Test.pth')
write_log("✅ Đã hoàn thành và lưu mô hình.")