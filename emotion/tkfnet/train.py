import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
from tqdm import tqdm
from model import TKFNet  # Import class TKFNet từ file model.py đã viết trước đó
from data_train import get_dataloaders

def train_model():
    # 1. Cấu hình thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    # 2. Khởi tạo Model, Loss và Dữ liệu
    model = TKFNet(num_classes=7).to(device)
    train_loader, test_loader = get_dataloaders(data_path="./data/raf-db", batch_size=128)
    
    criterion = nn.CrossEntropyLoss()
    
    # 3. Optimizer Momentum (LR=0.1 theo bài báo)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # 4. Polynomial Decay LR Scheduler (60 Epochs)
    # Giảm từ 0.1 xuống 0.01 với power 0.5
    scheduler = PolynomialLR(optimizer, total_iters=60, power=0.5)

    # 5. Vòng lặp huấn luyện
    num_epochs = 60
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct/total:.2f}%"})

        # Cập nhật Learning Rate sau mỗi epoch
        scheduler.step()

        # 6. Đánh giá trên tập Test
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total
        print(f"🟢 Test Accuracy: {test_acc:.2f}%")

        # Lưu model tốt nhất
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_tkfnet.pth")
            print("💾 Saved Best Model!")

    print(f"✅ Huấn luyện hoàn tất! Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()