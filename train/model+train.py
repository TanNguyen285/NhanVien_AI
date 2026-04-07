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
import copy

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ (HYPERPARAMETERS)
# ==========================================
NUM_CLASSES = 8
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {DEVICE}")

TRAIN_EPOCH = 15
TRAIN_LR = 1e-3
TRAIN_ES_PATIENCE = 4
TRAIN_DROPOUT = 0.2

FT_EPOCH = 35
FT_LR = 1e-5
FT_ES_PATIENCE = 10
FT_DROPOUT = 0.3

# ==========================================
# 2. CHUẨN BỊ DỮ LIỆU
# ==========================================
class EmotionDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.y[idx]

print("Loading data from h5 file...")
with h5py.File('my_custom_data.h5', 'r') as hf: 
    X_train, y_train = hf['X_train'][:], hf['y_train'][:]
    X_valid, y_valid = hf['X_val'][:], hf['y_val'][:]
    X_test,  y_test  = hf['X_test'][:], hf['y_test'][:]

X_train, y_train = shuffle(X_train, y_train, random_state=42)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(EmotionDataset(X_train, y_train, train_transforms), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(EmotionDataset(X_valid, y_valid, val_transforms), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(EmotionDataset(X_test, y_test, val_transforms), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. KIẾN TRÚC MÔ HÌNH (CHUẨN MOBILENETV2)
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
        
        # Load MobileNetV2 Pre-trained (V1 không còn bản chính chủ)
        # Lấy đến layer index 14 để có feature map 14x14, 96 channels
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

model = PAttLite(num_classes=NUM_CLASSES, dropout_rate=TRAIN_DROPOUT).to(DEVICE)

# ==========================================
# 4. HÀM HUẤN LUYỆN (Y CHANG LOGIC CỦA BẠN)
# ==========================================
def train_model(model, optimizer, criterion, epochs, patience, phase_name=""):
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        train_acc = running_corrects.double() / len(train_loader.dataset)
        
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"[{phase_name}] Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("=> Early Stopping!")
                break
                
    model.load_state_dict(best_model_wts)
    return model

# ==========================================
# 5. EXECUTION (2 PHASES)
# ==========================================
print("\n=== PHASE 1: WARM-UP ===")
for param in model.backbone.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer_p1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=TRAIN_LR)
model = train_model(model, optimizer_p1, criterion, TRAIN_EPOCH, TRAIN_ES_PATIENCE, "Phase 1")

print("\n=== PHASE 2: FINE-TUNING ===")
model.dropout.p = FT_DROPOUT 
for param in model.parameters():
    param.requires_grad = True

# Đóng băng BatchNorm (Mẹo của bạn)
for m in model.backbone.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
        for p in m.parameters(): p.requires_grad = False

optimizer_p2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FT_LR)
model = train_model(model, optimizer_p2, criterion, FT_EPOCH, FT_ES_PATIENCE, "Phase 2")

# ==========================================
# 6. TEST & SAVE
# ==========================================
model.eval()
test_corrects = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_corrects += torch.sum(preds == labels.data)

print(f"\nFinal Test Acc: {test_corrects.double()/len(test_loader.dataset)*100:.2f}%")
torch.save(model.state_dict(), 'PAtt_Lite_V2_PyTorch.pth')