import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_loaders_affectnet(data_dir, batch_size=64, img_size=224):
    """
    Hàm load dữ liệu AffectNet (RGB).
    Đã sửa lỗi đồng nhất biến và tránh Circular Import.
    """
    
    # --- 1. TĂNG CƯỜNG DỮ LIỆU (AUGMENTATION) ---
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 2. KIỂM TRA ĐƯỜNG DẪN THƯ MỤC ---
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    # Chốt chặn 1: Nếu không thấy folder 'train'
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy folder 'train' tại: {train_path}")

    # Chốt chặn 2: Nếu không thấy 'val', thử tìm 'test'
    if not os.path.exists(val_path):
        val_path = os.path.join(data_dir, 'test')
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Lỗi: Không tìm thấy folder 'val' hoặc 'test' tại: {data_dir}")

    # --- 3. LOAD DATASET ---
    try:
        train_set = datasets.ImageFolder(root=train_path, transform=transform_train)
        val_set = datasets.ImageFolder(root=val_path, transform=transform_val)
    except Exception as e:
        raise Exception(f"Lỗi khi đọc folder ảnh: {e}. Đảm bảo cấu trúc là: data_dir/train/class_name/anh.jpg")

    # --- 4. LẤY THÔNG TIN LỚP ---
    num_classes = len(train_set.classes)
    
    print("-" * 40)
    print(f"[INFO] Bộ dữ liệu: AffectNet RGB")
    print(f"[INFO] Đường dẫn: {data_dir}")
    print(f"[INFO] Cảm xúc tìm thấy ({num_classes} lớp): {train_set.classes}")
    print(f"[INFO] Tổng ảnh Train: {len(train_set)} | Val: {len(val_set)}")
    print("-" * 40)

    # --- 5. KHỞI TẠO DATALOADER ---
    # Ghi chú: num_workers nên để 4 cho ổn định trên Windows, 8 cho Linux/Server
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True 
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, num_classes