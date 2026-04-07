import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=64, img_size=48, num_workers=8):
    """
    Hàm chuẩn bị DataLoader cho nhận diện cảm xúc khuôn mặt (FER).
    - img_size: FER-2013 gốc là 48, nhưng nếu dùng Transfer Learning hãy đổi thành 224.
    - batch_size: FER có lượng ảnh lớn, nên để 64 hoặc 128 nếu GPU khỏe.
    """
    
    # 1. Augmentation cho FER (Tập trung vào biến đổi gương mặt)
    train_transforms = transforms.Compose([
        # Nếu dùng model pre-trained (ResNet/MobileNet), ảnh PHẢI là 3 kênh RGB
        # Grayscale(3) biến 1 kênh xám thành 3 kênh (R=G=B) để khớp input model
        transforms.Grayscale(num_output_channels=3), 
        transforms.Resize((img_size, img_size)),
        
        # Augmentation nhẹ nhàng để không làm biến dạng đặc điểm cảm xúc
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10), # Xoay nhẹ đầu
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Thay đổi ánh sáng nhẹ
        
        transforms.ToTensor(),
        # Normalize theo thông số chuẩn của ImageNet (giúp model hội tụ nhanh hơn)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Tập Val/Test: Giữ nguyên để đánh giá chính xác
    val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. Đường dẫn thư mục (Cấu trúc: data_dir/train/angry, data_dir/test/angry...)
    # Lưu ý: FER-2013 trên Kaggle thường chia folder là 'train' và 'test' thay vì 'val'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val') # Hoặc 'val' tùy cách bạn đặt tên folder

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"LỖI: Kiểm tra lại đường dẫn {data_dir}. Phải có folder 'train' và 'val'!")

    # Load dữ liệu bằng ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    # 4. Tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)

    # 5. Báo cáo kết quả load
    print(f"[*] Đã sẵn sàng bộ dữ liệu Emotion!")
    print(f"    - Tổng số ảnh Train: {len(train_dataset)}")
    print(f"    - Tổng số ảnh Test/Val: {len(val_dataset)}")
    print(f"    - Các lớp cảm xúc: {list(train_dataset.class_to_idx.keys())}")
    
    return train_loader, val_loader, train_dataset.class_to_idx

# ==========================================
# TEST THỬ CODE
# ==========================================
if __name__ == "__main__":
    # ĐƯỜNG DẪN: Trỏ vào folder chứa 2 folder con là 'train' và 'test'
    PATH_DATA = './fer2013' 
    
    try:
        # Nếu bạn dùng model tự xây (Simple CNN): img_size=48
        # Nếu bạn dùng ResNet/MobileNet: img_size=224
        train_loader, val_loader, labels = get_data_loaders(PATH_DATA, batch_size=32, img_size=48)
        
        # Check thử 1 batch
        imgs, lbls = next(iter(train_loader))
        print(f"\n[OK] Shape của batch ảnh: {imgs.shape}") # [32, 3, 48, 48]
        
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")