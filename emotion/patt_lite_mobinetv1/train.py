import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import time
import os

# Import từ file của bạn
from model import PAttLite_V1_Final
from emotion.patt_lite_mobinetv1.data_loader import get_data_loaders

# ==========================================
# CẤU HÌNH SIÊU THAM SỐ (OPTIMIZED)
# ==========================================
CONFIG = {
    "num_classes": 8,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "data_dir": "C:\\Users\\ThisPC\\Desktop\\data_emo\\RAF-DB_lite\\Train", # Đã sửa path
    
    #"data_dir": "C:/Users/ThisPC/Desktop/data_emo/data_emo_new", # Đã sửa path
    "batch_size": 32, # Tối ưu cho 40k ảnh 224x224
    "phase1": {"lr": 1e-3, "epochs": 15}, # 40k ảnh chỉ cần 15 epochs để hội tụ head
    "phase2": {"lr": 5e-6, "epochs": 35, "unfreeze": 7}, # LR cực thấp cho fine-tune
    "save_name": "pattlite_emotion_final.pth"
}

def train():
    # 0. Khởi tạo Data và Model
    # Đảm bảo hàm get_data_loaders trong file của bạn nhận batch_size từ CONFIG
   # Trong train.py
    train_loader, val_loader, test_loader = get_data_loaders(data_dir=CONFIG["data_dir"], batch_size=CONFIG["batch_size"])
    
    model = PAttLite_V1_Final(num_classes=CONFIG["num_classes"]).to(CONFIG["device"])
    
    # Sử dụng Label Smoothing vì ảnh 75x75 upscale lên 224x224 sẽ bị mờ, 
    # giúp model không quá tự tin vào các pixel bị nhòe.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    # =========================================================
    # PHASE 1: TRAIN HEAD (BACKBONE FROZEN)
    # =========================================================
    print(f"\n🚀 PHASE 1: Warm-up Head ({CONFIG['phase1']['epochs']} Epochs)")
    
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["phase1"]["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    best_val_acc = 0.0
    
    for epoch in range(CONFIG["phase1"]["epochs"]):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        v_loss, v_acc = evaluate(model, val_loader, criterion)
        
        scheduler.step(v_loss)
        print(f"E{epoch+1:02d} | Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}% | Val Loss: {v_loss:.4f}")
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), "temp_head.pth")

    # Dọn dẹp bộ nhớ trước khi sang Phase 2
    torch.cuda.empty_cache()

    # =========================================================
    # PHASE 2: FINE-TUNING (UNFREEZE PARTIAL BACKBONE)
    # =========================================================
    print(f"\n🔥 PHASE 2: Fine-tuning Backbone ({CONFIG['phase2']['epochs']} Epochs)")
    
    model.load_state_dict(torch.load("temp_head.pth"))
    
    # Mở khóa các lớp cuối của backbone
    children = list(model.backbone.children())
    num_children = len(children)
    for i, child in enumerate(children):
        if i >= (num_children - CONFIG["phase2"]["unfreeze"]):
            for param in child.parameters():
                if not isinstance(child, nn.BatchNorm2d): # Giữ BN frozen để ổn định
                    param.requires_grad = True

    # Sử dụng AdamW cho fine-tuning vì có weight_decay tốt hơn
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=CONFIG["phase2"]["lr"], weight_decay=0.01)
    
    # Dùng CosineAnnealing cho 40k ảnh giúp model hội tụ sâu hơn
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["phase2"]["epochs"])

    for epoch in range(CONFIG["phase2"]["epochs"]):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        v_loss, v_acc = evaluate(model, val_loader, criterion)
        
        scheduler.step()
        
        print(f"FT-E{epoch+1:02d} | Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.8f}")
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), CONFIG["save_name"])
            print(f"✅ Đã lưu Model tốt nhất: {v_acc:.2f}%")

# --- HÀM TRỢ GIÚP ---

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    running_loss, correct, total = 0, 0, 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        
        # Gradient Clipping (Global Clipnorm = 3.0)
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss/len(loader), 100. * correct / total

def evaluate(model, loader, criterion):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return val_loss/len(loader), 100. * correct / total

if __name__ == "__main__":
    train()