import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datetime import datetime

from data_loader import make_loader, get_class_weights
from Pattlite import PAttLite

# ==========================================
# CẤU HÌNH
# ==========================================
DATA_ROOT   = r"C:\Users\ThisPC\Documents\GitHub\NhanVien_AI_Emo\dataset"
NUM_CLASSES = 7
BATCH_SIZE  = 32
IMAGE_SIZE  = 112
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Angry', 'Contempt', 'Disgust', 'Happy', 'Neutral', 'Sad', 'Surprise']
LOG_FILE    = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


# ==========================================
# LOGGING
# ==========================================
def write_log(message: str):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


# ==========================================
# CONFUSION MATRIX
# ==========================================
def plot_confusion_matrix(model, loader, phase_name: str):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix - {phase_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"cm_{phase_name}.png")
    plt.close()
    write_log(f"  Đã lưu confusion matrix: cm_{phase_name}.png")


# ==========================================
# TRAIN LOOP
# ==========================================
def train_model(model, optimizer, criterion, train_loader, val_loader,
                epochs: int, patience: int, phase_name: str):
    best_val_acc   = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve     = 0

    for epoch in range(epochs):
        # --- Train ---
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

        # --- Validate ---
        model.eval()
        val_corr = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corr += torch.sum(preds == labels.data)

        val_acc = val_corr.double() / len(val_loader.dataset)

        write_log(f"[{phase_name}] Epoch {epoch+1:02d} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve     = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                write_log(f"=> Early Stopping tại epoch {epoch+1}!")
                break

    model.load_state_dict(best_model_wts)
    write_log(f"=> Best Val Acc [{phase_name}]: {best_val_acc:.4f}")
    return model


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # --- Load data ---
    write_log("📊 Đang tải dữ liệu từ thư mục ảnh...")
    train_loader = make_loader(DATA_ROOT, 'train', IMAGE_SIZE, BATCH_SIZE,
                               augment=True,  write_log=write_log)
    val_loader   = make_loader(DATA_ROOT, 'val',   IMAGE_SIZE, BATCH_SIZE,
                               augment=False, write_log=write_log)
    test_loader  = make_loader(DATA_ROOT, 'test',  IMAGE_SIZE, BATCH_SIZE,
                               augment=False, write_log=write_log)

    class_weights_tensor = get_class_weights(train_loader, DEVICE, write_log)

    # --- Khởi tạo model ---
    model     = PAttLite(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    write_log(f"\n🖥️  Device: {DEVICE}")
    write_log(f"📁 Data root: {DATA_ROOT}")

    # --- Phase 1: Warm-up (đóng băng backbone) ---
    write_log("\n=== PHASE 1: WARM-UP (frozen backbone) ===")
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    model = train_model(model, optimizer, criterion,
                        train_loader, val_loader,
                        epochs=15, patience=4, phase_name="Phase 1")

    # --- Phase 2: Fine-tuning (mở toàn bộ, giữ BN frozen) ---
    write_log("\n=== PHASE 2: FINE-TUNING (full model) ===")
    for param in model.parameters():
        param.requires_grad = True
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
    )
    model = train_model(model, optimizer, criterion,
                        train_loader, val_loader,
                        epochs=35, patience=10, phase_name="Phase 2")

    # --- Kết quả cuối ---
    write_log("\n🎨 Đang tạo Confusion Matrix cho tập Test...")
    plot_confusion_matrix(model, test_loader, "Final_Test")

    torch.save(model.state_dict(), "PAtt_Lite_final.pth")
    write_log("✅ Hoàn thành! Mô hình đã lưu: PAtt_Lite_final.pth")