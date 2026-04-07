import torch
import torch.optim as optim
from timm.loss import LabelSmoothingCrossEntropy
from model import MicroViT 
from data_loader import get_loaders_affectnet
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CẤU HÌNH
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50           # Tổng số epoch bạn muốn đạt tới
BATCH_SIZE = 32
DATA_DIR = './data_train' 
CHECKPOINT_DIR = 'checkpoints'
LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pth')
HISTORY_PATH = 'training_history.pth'

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Khởi tạo Data và Model gốc
train_loader, val_loader, n_classes = get_loaders_affectnet(DATA_DIR, batch_size=BATCH_SIZE)
model = MicroViT(variant='S2', num_classes=n_classes).to(device)

# Khởi tạo Optimizer & Scheduler gốc
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Biến bổ trợ
start_epoch = 1
history = {'loss': [], 'acc': []}
best_acc = 0.0

# ==========================================
# 2. TỰ ĐỘNG CHECK & LOAD FILE CŨ (RESUME)
# ==========================================
if os.path.exists(LAST_CKPT_PATH) and os.path.exists(HISTORY_PATH):
    print(f"\n[v] Tìm thấy file cũ. Đang nạp dữ liệu để chạy tiếp...")
    try:
        checkpoint = torch.load(LAST_CKPT_PATH, map_location=device)
        
        # Nạp trọng số Model & Optimizer
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Nạp số Epoch và Lịch sử
        start_epoch = checkpoint['epoch'] + 1
        history = torch.load(HISTORY_PATH)
        best_acc = max(history['acc']) if history['acc'] else 0.0
        
        # Quan trọng: Đưa Scheduler về đúng nhịp LR cũ
        for _ in range(checkpoint['epoch']):
            scheduler.step()
            
        print(f"--> Tiếp tục từ Epoch {start_epoch}. LR hiện tại: {optimizer.param_groups[0]['lr']:.8f}")
    except Exception as e:
        print(f"[!] Lỗi khi load file cũ: {e}. Sẽ chạy lại từ đầu.")
else:
    print("\n[x] Không thấy checkpoint cũ. Bắt đầu huấn luyện mới từ Epoch 1.")

# ==========================================
# 3. HÀM VẼ BIỂU ĐỒ
# ==========================================
def auto_draw_report(history_data):
    sns.set_theme(style="whitegrid")
    epochs = range(1, len(history_data['loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, history_data['loss'], color='tab:red', label='Train Loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:blue')
    ax2.plot(epochs, history_data['acc'], color='tab:blue', label='Val Accuracy')
    plt.title('MicroViT Training Progress')
    plt.savefig('live_report.png')
    plt.close()

# ==========================================
# 4. VÒNG LẶP HUẤN LUYỆN
# ==========================================
if __name__ == "__main__":
    for epoch in range(start_epoch, EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)

        # --- VALIDATE ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, pred = outputs.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
        
        epoch_acc = 100. * correct / total
        
        # Cập nhật LR theo nhịp Cosine
        scheduler.step()

        # --- LƯU TRỮ & GHI ĐÈ ---
        history['loss'].append(avg_loss)
        history['acc'].append(epoch_acc)
        torch.save(history, HISTORY_PATH)
        auto_draw_report(history)
        
        # Lưu checkpoint "Cuối cùng" để lần sau Resume
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, LAST_CKPT_PATH)

        # Lưu checkpoint "Tốt nhất" để đem đi Test (Inference)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_microvit.pth'))
            status = "MỚI NHẤT!"
        else:
            status = ""

        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {epoch_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.8f} {status}")

    print(f"--- Hoàn thành! Best Acc: {best_acc:.2f}% ---")