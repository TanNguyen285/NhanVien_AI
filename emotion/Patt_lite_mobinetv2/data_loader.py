import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

# 1. Lớp chuyển đổi tùy chỉnh để thích ứng với mọi loại ảnh (Xám hoặc Màu)
class EnsureRGB(object):
    """Tự động chuyển đổi sang RGB nếu ảnh là Grayscale, giữ nguyên nếu đã là RGB."""
    def __call__(self, img):
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img

# 2. Lớp bọc Subset để áp dụng Transform riêng biệt cho Train/Val/Test sau khi split
class ApplyTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_data_loaders(data_dir, batch_size=64, img_size=112, split_ratio=(0.8, 0.1, 0.1)):
    """
    Hàm đọc dữ liệu thông minh:
    - Tự động nhận diện cấu trúc folder (train/val/test hoặc folder tổng).
    - Tự động chuyển đổi ảnh xám/màu về 3 kênh cho MobileNet.
    - Áp dụng chuẩn hóa theo ImageNet.
    """
    
    # Thông số chuẩn của ImageNet
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # Transform cho Train (Có Augmentation)
    train_transform = transforms.Compose([
        EnsureRGB(),
        transforms.Resize((img_size, img_size)), # Resize cố định trước
    # Thay RandomResizedCrop bằng Affine để xoay/dịch chuyển mà không mất mặt
    transforms.RandomAffine(
        degrees=15, 
        translate=(0.1, 0.1), 
        scale=(0.9, 1.1), 
        shear=5
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Giữ nguyên vì rất tốt
    transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Transform cho Validation & Test (Chỉ Resize & Normalize)
    test_transform = transforms.Compose([
        EnsureRGB(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Kiểm tra sự tồn tại của thư mục
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Thư mục dữ liệu không tồn tại: {data_dir}")

    # Kiểm tra cấu trúc thư mục
    sub_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    has_split = all(d in sub_dirs for d in ['train', 'val', 'test'])

    if has_split:
        print("📁 Phát hiện cấu trúc train/val/test có sẵn.")
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)
    else:
        print("📂 Không tìm thấy train/val/test. Tiến hành tự động phân tách dữ liệu...")
        full_dataset = datasets.ImageFolder(data_dir)
        total_size = len(full_dataset)
        
        if total_size == 0:
            raise ValueError(f"❌ Thư mục {data_dir} không chứa ảnh hợp lệ trong các thư mục con!")

        # Tính toán kích thước (Đã fix lỗi NameError split_ratio)
        train_len = int(split_ratio[0] * total_size)
        val_len = int(split_ratio[1] * total_size)
        test_len = total_size - train_len - val_len
        
        # Split dữ liệu
        train_data, val_data, test_data = random_split(
            full_dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Áp dụng Transform riêng biệt
        train_dataset = ApplyTransform(train_data, transform=train_transform)
        val_dataset = ApplyTransform(val_data, transform=test_transform)
        test_dataset = ApplyTransform(test_data, transform=test_transform)
        
        # Lưu thông tin nhãn lớp
        train_dataset.classes = full_dataset.classes

    # Cấu hình DataLoaders
    # Giảm num_workers nếu chạy trên Windows gặp lỗi Multiprocessing
    num_workers = 4 if os.name != 'nt' else 0 

    common_params = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True if torch.cuda.is_available() else False
    }

    train_loader = DataLoader(train_dataset,
                               shuffle=True, 
                               prefetch_factor=2 if num_workers > 0 else None,
                                drop_last=True,
                                 **common_params)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_params)

    print(f"✅ Setup thành công!")
    print(f"📊 Thống kê: Train({len(train_dataset)}), Val({len(val_dataset)}), Test({len(test_dataset)})")
    print(f"🏷️  Classes: {getattr(train_dataset, 'classes', 'Unknown')}")
    
    return train_loader, val_loader, test_loader