import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from model import PAttLite

# ==========================================
# CONFIG
# ==========================================
NUM_CLASSES = 7
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TRAIN_EPOCH = 10  # Tăng lên một chút vì ảnh xám cần học kỹ hơn
TRAIN_LR = 1e-3
TRAIN_ES_PATIENCE = 5

FT_EPOCH = 20
FT_LR = 1e-5
FT_ES_PATIENCE = 7

# Đảm bảo CLASS_NAMES khớp với thứ tự folder trong tập data của bạn
CLASS_NAMES = ['angry','contempt','disgust', 'happy', 'neutral', 'sad', 'surprise']

# ==========================================
# DATASET
# ==========================================
class EmotionDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx] # Ảnh từ h5 đang là (112, 112, 1)
        if self.transform:
            img = self.transform(img)
        return img, self.y[idx]

print("Loading grayscale data (112x112)...")
# CẬP NHẬT TÊN FILE H5 CỦA BẠN Ở ĐÂY
H5_PATH = 'emodata_112x112_gray.h5' 

with h5py.File(H5_PATH, 'r') as hf:
    X_train, y_train = hf['X_train'][:], hf['y_train'][:]
    X_valid, y_valid = hf['X_val'][:], hf['y_val'][:]
    X_test,  y_test  = hf['X_test'][:], hf['y_test'][:]

X_train, y_train = shuffle(X_train, y_train, random_state=42)

# ==========================================
# TRANSFORMS (Quan trọng nhất cho ảnh Gray)
# ==========================================
# Vì model backbone thường đợi ảnh 3 kênh, ta convert Gray -> RGB (3 kênh giống nhau)
common_transforms = [
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3), # Biến (112,112,1) thành (112,112,3)
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose(common_transforms)

# DataLoader giữ nguyên phần Sampler của bạn (rất tốt cho imbalance)
class_sample_count = np.bincount(y_train)
weights = 1. / class_sample_count
samples_weight = weights[y_train]
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_loader = DataLoader(EmotionDataset(X_train, y_train, train_transforms), batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(EmotionDataset(X_valid, y_valid, val_transforms), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(EmotionDataset(X_test, y_test, val_transforms), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# LOSS & WEIGHTS
# ==========================================
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

# ==========================================
# TRAIN FUNCTION (Giữ nguyên logic của bạn)
# ==========================================
def train_model(model, optimizer, scheduler, criterion, epochs, patience, phase_name=""):
    best_val_acc = 0.0
    epochs_no_improve = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_correct += torch.sum(preds == labels)

        train_acc = train_correct.double() / len(train_loader.dataset)
        
        # Validation logic
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels)
        
        val_acc = val_correct.double() / len(val_loader.dataset)
        print(f"[{phase_name}] Epoch {epoch+1:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if scheduler: scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_{phase_name}.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break
    return model

# ==========================================
# EXECUTION
# ==========================================
model = PAttLite(num_classes=NUM_CLASSES).to(DEVICE)

# PHASE 1: Warm up (Freeze backbone)
print("\n--- PHASE 1: WARM UP ---")
for p in model.backbone.parameters(): p.requires_grad = False
optimizer_p1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=TRAIN_LR)
model = train_model(model, optimizer_p1, None, criterion, TRAIN_EPOCH, TRAIN_ES_PATIENCE, "phase1")

# PHASE 2: Fine-tuning (Unfreeze)
print("\n--- PHASE 2: FINE TUNING ---")
for p in model.parameters(): p.requires_grad = True
# Giữ BatchNorm ở trạng thái eval để ổn định
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d): m.eval()

optimizer_p2 = optim.Adam([
    {"params": model.backbone.parameters(), "lr": FT_LR},
    {"params": model.classifier.parameters(), "lr": FT_LR * 10} # Classifier học nhanh hơn
])
scheduler_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_p2, mode='max', patience=2, factor=0.5)

model = train_model(model, optimizer_p2, scheduler_p2, criterion, FT_EPOCH, FT_ES_PATIENCE, "phase2")

# ==========================================
# FINAL TEST & CONFUSION MATRIX
# ==========================================
print("\n--- FINAL TESTING ---")
model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_phase2.pth"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix - Grayscale 112x112")
plt.savefig("confusion_matrix.png")
torch.save(model.state_dict(), "PAttLite_emodata.pth")
print("ALL DONE!")