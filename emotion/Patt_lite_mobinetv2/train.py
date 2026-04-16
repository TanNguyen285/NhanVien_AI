import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import h5py # Thêm thư viện đọc h5
from torch.utils.data import Dataset, DataLoader # Thêm Dataset
import numpy as np

# Import từ các file module của bạn
from model import LightAttNet 
from data_loader import get_data_loaders

# =========================================================
# 1. CẤU HÌNH HỆ THỐNG & SIÊU THAM SỐ
# =========================================================
CONFIG = {
    "num_classes": 8,
    "img_size": 112,
    "batch_size": 32,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #"data_dir": "C:/Users/ThisPC/Desktop/data_emo/emooo/data",
    "h5_path": "C:/Users/ThisPC/Documents/GitHub/DATA_Emotion/emodata_112x112_gray.h5", # Đường dẫn file H5
    "use_h5": True, # Cờ để chọn đọc từ folder hay file H5
    
    # Phase 1: Huấn luyện tầng Attention & Classifier
    "p1_epochs": 15,
    "p1_lr": 1e-3,
    
    # Phase 2: Fine-tuning sâu vào các khối Inverted Residual của V2
    "p2_epochs": 35,
    "p2_lr": 1e-5,
    "unfreeze_blocks": 5, 
    
    "model_path": "Patt_lite_mobinetv2_model.pth"
}

# --- PHẦN BỔ SUNG: CLASS ĐỌC H5 ---
class H5EmotionDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import transforms # Thêm transforms để xử lý ảnh từ H5

def save_log(message, file_path="training_log_mobinetv2.txt"):
    """Ghi lại tiến trình vào file txt"""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def plot_confusion_matrix(model, loader, device, classes, save_path="confusion_matrix.png"):
    """Vẽ và lưu biểu đồ ma trận nhầm lẫn giữa các lớp"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Thực tế (Actual)')
    plt.xlabel('Dự đoán (Predicted)')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Đã lưu ma trận nhầm lẫn tại: {save_path}")

def train():
    # 2. Khởi tạo Data Loaders (Sửa đổi để hỗ trợ cả H5)
    if CONFIG["use_h5"] and os.path.exists(CONFIG["h5_path"]):
        print(f"📂 Đang tải dữ liệu từ file H5: {CONFIG['h5_path']}")
        
        # Định nghĩa transform cho dữ liệu H5 (ảnh xám -> 3 kênh màu)
        train_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        with h5py.File(CONFIG["h5_path"], 'r') as hf:
            train_loader = DataLoader(H5EmotionDataset(hf['X_train'][:], hf['y_train'][:], train_tf), 
                                      batch_size=CONFIG["batch_size"], shuffle=True)
            val_loader = DataLoader(H5EmotionDataset(hf['X_val'][:], hf['y_val'][:], val_tf), 
                                    batch_size=CONFIG["batch_size"])
            test_loader = DataLoader(H5EmotionDataset(hf['X_test'][:], hf['y_test'][:], val_tf), 
                                     batch_size=CONFIG["batch_size"])
    else:
        # Giữ nguyên logic cũ của folder
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir=CONFIG["data_dir"], 
            batch_size=CONFIG["batch_size"],
            img_size=CONFIG["img_size"]
        )

    # 3. Khởi tạo Model (Sử dụng tên mới LightAttNet)
    model = LightAttNet(num_classes=CONFIG["num_classes"]).to(CONFIG["device"])
    
    # Label Smoothing giúp model giảm "overconfidence" khi ảnh bị mờ/nhiễu
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler() # Hỗ trợ Mixed Precision training

    # =========================================================
    # PHASE 1: WARM-UP (Đóng băng toàn bộ Backbone)
    # =========================================================
    print(f"\n🚀 PHASE 1: Warm-up Head ({CONFIG['p1_epochs']} Epochs)")
    
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["p1_lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    best_val_acc = 0.0

    for epoch in range(CONFIG["p1_epochs"]):
        t_loss, t_acc = run_epoch(model, train_loader, optimizer, criterion, scaler, is_training=True)
        v_loss, v_acc = run_epoch(model, val_loader, None, criterion, None, is_training=False)
        log_msg = f"[phase1] Epoch {epoch+1:02d} | Train Acc: {t_acc/100:.4f} | Val Acc: {v_acc/100:.4f}"
        print(log_msg)
        save_log(log_msg)
        scheduler.step(v_loss)
        print(f"E{epoch+1:02d} | Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}% | Val Loss: {v_loss:.4f}")
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), "temp_v2_head.pth")

    # =========================================================
    # PHASE 2: FINE-TUNING (Mở khóa các khối Inverted Residual cuối)
    # =========================================================
    print(f"\n🔥 PHASE 2: Fine-tuning Inverted Residuals ({CONFIG['p2_epochs']} Epochs)")
    
    model.load_state_dict(torch.load("temp_v2_head.pth")) # Đã sửa tên file load cho khớp temp
    
    # Mở khóa các tầng cuối của MobileNetV2
    backbone_layers = list(model.backbone.children())
    num_layers = len(backbone_layers)
    for i, layer in enumerate(backbone_layers):
        if i >= (num_layers - CONFIG["unfreeze_blocks"]):
            for param in layer.parameters():
                if not isinstance(layer, nn.BatchNorm2d):
                    param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=CONFIG["p2_lr"], weight_decay=0.05)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["p2_epochs"])

    for epoch in range(CONFIG["p2_epochs"]):
        t_loss, t_acc = run_epoch(model, train_loader, optimizer, criterion, scaler, is_training=True)
        v_loss, v_acc = run_epoch(model, val_loader, None, criterion, None, is_training=False)
        log_msg = f"[phase2] Epoch {epoch+1:02d} | Train Acc: {t_acc/100:.4f} | Val Acc: {v_acc/100:.4f}"
        print(log_msg)
        save_log(log_msg)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"FT-E{epoch+1:02d} | Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}% | LR: {current_lr:.8f}")
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), CONFIG["model_path"])
            print(f"🌟 New Record: {v_acc:.2f}% - Model Saved!")

    # Sau training, vẽ Confusion Matrix
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
    plot_confusion_matrix(model, test_loader, CONFIG["device"], class_names)

# =========================================================
# 4. HÀM CHẠY EPOCH (HỢP NHẤT TRAIN/VAL)
# =========================================================
def run_epoch(model, loader, optimizer, criterion, scaler, is_training=False):
    if is_training:
        model.train()
    else:
        model.eval()

    running_loss, correct, total = 0, 0, 0
    device = CONFIG["device"]

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if is_training:
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss/len(loader), 100. * correct / total

if __name__ == "__main__":
    train()